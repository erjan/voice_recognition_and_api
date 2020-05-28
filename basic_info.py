from pydub import AudioSegment
import sys
import wave
import librosa

from sox import file_info

if __name__=="__main__":

    #voice = AudioSegment.from_wav('jaks.2.wav')
    #new_voice = voice[0:10000]
    #new_voice.export('newjaks.wav', format='wav')

    
    print(sys.argv)
    first_dur = librosa.get_duration(filename = sys.argv[1])
    print('file duration : %d ' % first_dur)

    f = wave.open(sys.argv[1])
    print('num of channels %d' % f.getnchannels())



'''
    file1  = sys.argv[1]

    ch = file_info.channels(file1)
    print('for file: ' + file1)
    print('num of channels is: %d' % ch)
    '''
    #return jsonify({'response': 'num of channels in file 1: ' + str(ch) })
