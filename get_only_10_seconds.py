import sys
from pydub import AudioSegment

def get_customer_voice_10_seconds(file):
    voice = AudioSegment.from_wav(file)
    new_voice = voice[0:10000]
    file = str(file) + '_10seconds.wav'
    new_voice.export(file, format='wav')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('give wav file to process!')
    else:
        print(sys.argv)
        get_customer_voice_10_seconds(sys.argv[1])
