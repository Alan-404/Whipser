from preprocess.text import TextProcessor
from preprocess.audio import AudioProcessor
import pandas as pd
import pickle

def save_data(path: str, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def preprocessing_data(data_path: str,
                       save_audio_path: str,
                       save_text_path: str,
                       tokenizer_path: str,
                       n_mel_channels: int=80,
                       sample_rate: int = 16000,
                       duration: int = 15,
                       frame_size: int = int(16000*0.025),
                       hop_length: int = int(16000*0.01)):
    audio_processor = AudioProcessor(
        sample_rate=sample_rate,
        duration=duration,
        mono=True,
        frame_size=frame_size,
        hop_length=hop_length,
        n_mels=n_mel_channels
    )
    text_processor = TextProcessor(tokenizer_path=tokenizer_path)

    df = pd.read_csv(data_path)
    audio_data = audio_processor.process(df['audio'].to_list())
    text_data = text_processor.process(df['label'].to_list())

    save_data(save_audio_path, audio_data)
    save_data(save_text_path, text_data)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--text_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--frame_size", type=int, default=None)
    parser.add_argument("--hop_length", type=int, default=None)

    args = parser.parse_args()

    if args.frame_size is None:
        args.frame_size = int(args.sample_rate * 0.025)
    if args.hop_length is None:
        args.frame_size = int(args.sample_rate * 0.01)

    preprocessing_data(
        data_path=args.data_path,
        save_audio_path=args.audio_path,
        save_text_path=args.text_path,
        tokenizer_path=args.tokenizer_path,
        n_mel_channels=args.n_mel_channels,
        sample_rate=args.sample_rate,
        duration=args.duration,
        frame_size=args.frame_size,
        hop_length=args.hop_length
    )