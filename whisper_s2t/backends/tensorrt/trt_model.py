import json
import torch
import tensorrt_llm

from pathlib import Path
from collections import OrderedDict

from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config

def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=0):
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, seq_len, feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return output_tensor


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config('encoder', engine_dir)
        self.n_mels = config['n_mels']
        self.dtype = config['dtype']
        self.num_languages = config['num_languages']
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = engine_dir / 'encoder' / 'rank0.engine'
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_audio_features(self, mel):
        # Input_lengths here are actually encoder_output_lengths for whisper.
        # Since the conv subsampling layer in the whisper decoder, seq_len would divide by 2.
        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)
        encoder_max_input_length = torch.max(input_lengths).item()
        if self.encoder_config['plugin_config']['remove_input_padding']:
            mel_input_lengths = torch.full((mel.shape[0], ),
                                           mel.shape[2],
                                           dtype=torch.int32,
                                           device='cuda')
            # mel B,D,T -> B,T,D -> BxT, D
            mel = mel.transpose(1, 2)
            mel = remove_tensor_padding(mel, mel_input_lengths)

        inputs = OrderedDict()
        inputs['input_features'] = mel
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('input_features', str_dtype_to_trt(self.dtype),
                       mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['encoder_output']
        return audio_features

class WhisperDecoding:

    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            paged_kv_cache=self.decoder_config['plugin_config']
            ['paged_kv_cache'],
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
            gather_generation_logits=True,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def single_logits(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_input_lengths):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [batch_size, 1, encoder_max_input_length]).int().cuda()

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            1,
            beam_width=1,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID)
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full((encoder_outputs.shape[0], ),
                                                 encoder_outputs.shape[1],
                                                 dtype=torch.int32,
                                                 device='cuda')

                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lens)
        sampling_config = SamplingConfig(
           end_id=50257,
           pad_id=50257,
        )

        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
            return_dict=True,
            output_generation_logits=True,
        )
        torch.cuda.synchronize()

        return output_ids['generation_logits']

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 sampling_config,
                 encoder_max_input_length,
                 encoder_input_lengths):
        batch_size = decoder_input_ids.shape[0]
        decoder_input_lengths = torch.tensor([
            decoder_input_ids.shape[-1]
            for _ in range(decoder_input_ids.shape[0])
        ],
                                             dtype=torch.int32,
                                             device='cuda')
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones(
            [batch_size, 1, encoder_max_input_length]).int().cuda()

        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            sampling_config.max_new_tokens,
            beam_width=sampling_config.num_beams,
            encoder_max_input_length=encoder_max_input_length)

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            # 50256 is the index of <pad> for all whisper models' decoder
            WHISPER_PAD_TOKEN_ID = 50256
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=WHISPER_PAD_TOKEN_ID)
            if encoder_outputs.dim() == 3:
                encoder_output_lens = torch.full((encoder_outputs.shape[0], ),
                                                 encoder_outputs.shape[1],
                                                 dtype=torch.int32,
                                                 device='cuda')

                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_output_lens)
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRT(object):
    def __init__(self, engine_dir, compute_type='float16', debug_mode=False):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir,
                                       runtime_mapping,
                                       debug_mode=False)
        self.n_mels = self.encoder.n_mels
        self.is_multilingual = True
        self.compute_type = compute_type

    @torch.no_grad()
    def detect_language(
        self, mel: torch.Tensor, tokenizer,
    ):
        """
        Detect the spoken language in the audio, and return them as list of strings, along with the ids
        of the most probable language tokens and the probability distribution over all language tokens.
        This is performed outside the main decode loop in order to not interfere with kv-caching.

        Returns
        -------
        language_tokens : Tensor, shape = (n_audio,)
            ids of the most probable language tokens, which appears after the startoftranscript token.
        language_probs : List[Dict[str, float]], length = n_audio
            list of dictionaries containing the probability distribution over all languages.
        """
        single = mel.ndim == 2
        if single:
            mel = mel.unsqueeze(0)

        encoder_input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)
        encoder_max_input_length = torch.max(encoder_input_lengths).item()

        n_audio = mel.shape[0]

        # skip encoder forward pass if already-encoded audio features were given
        if mel.shape[1] == self.n_mels:
            mel = self.encode(mel)

        # forward pass using a single token, startoftranscript
        x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
        logits = self.decoder.single_logits(x, mel, encoder_max_input_length, encoder_input_lengths)[0]

        # collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(tokenizer.lang_code_to_token_id.values())] = False
        logits[:, mask] = -torch.inf
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item() for c, j in tokenizer.lang_code_to_token_id.items()
            }
            for i in range(n_audio)
        ]

        language_probs = language_probs[0]

        return language_probs

    def encode(self, mel):
        return self.encoder.get_audio_features(mel)

    def generate(self, features, prompts, **generate_kwargs):
        features = features.half()

        sampling_config = SamplingConfig(**generate_kwargs)

        encoder_input_lengths = torch.tensor(
            [features.shape[2] // 2 for _ in range(features.shape[0])],
            dtype=torch.int32,
            device=features.device)
        encoder_max_input_length = torch.max(encoder_input_lengths).item()

        if features.shape[1] == self.n_mels:
            features = self.encode(features)

        decoder_input_ids = torch.tensor(prompts)

        output_ids = self.decoder.generate(decoder_input_ids,
                                           features,
                                           sampling_config,
                                           encoder_max_input_length,
                                           encoder_input_lengths)

        return output_ids

