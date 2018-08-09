import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="file://distributed.dpt",
        cudnn_enabled=True,
        cudnn_benchmark=False,

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/train.txt',
        validation_files='filelists/test.txt',
        text_cleaners=['english_cleaners'],
        sort_by_length=False,

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,  # if None, half the sampling rate
        # from tacotron-gst
        num_freq=1025,
        sample_rate=16000,
        frame_length_ms=50,
        frame_shift_ms=12.5,
        preemphasis=0.97,
        min_level_db=-100,
        ref_level_db=20,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # GST Parameters
        style_token=10,
        num_heads=4,
        use_gst=True,
        linear_dim=1025,
        # Speaker Encoder
        style='',  # ['speaker_encoder','style_embedding','both','']
        num_mel=40,
        se_sample_rate =8000,
        se_window=0.025,
        se_hop=100,
        se_frame=180,
        se_n_fft=512,
        se_checkpoint='/home/zeng/work/pycharm/tacotron2/ge2e_model/checkpoint_step000007000.pth',
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.6,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1,
        batch_size=4,
        mask_padding=False  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
