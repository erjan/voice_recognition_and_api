import numpy
import scipy.io.wavfile as wf
import sys

print(sys.argv)

class VoiceActivityDetection:

    def __init__(self):
        self.__step = 160
        self.__buffer_size = 160
        self.__buffer = numpy.array([],dtype=numpy.int16)
        self.__out_buffer = numpy.array([],dtype=numpy.int16)
        self.__n = 0
        self.__VADthd = 0.
        self.__VADn = 0.
        self.__silence_counter = 0

    # Voice Activity Detection
    # Adaptive threshold
    def vad(self, _frame):
        frame = numpy.array(_frame) ** 2.
        result = True
        threshold = 0.1
        thd = numpy.min(frame) + numpy.ptp(frame) * threshold
        self.__VADthd = (self.__VADn * self.__VADthd + thd) / float(self.__VADn + 1.)
        self.__VADn += 1.

        if numpy.mean(frame) <= self.__VADthd:
            self.__silence_counter += 1
        else:
            self.__silence_counter = 0

        if self.__silence_counter > 20:
            result = False
        return result

    # Push new audio samples into the buffer.
    def add_samples(self, data):
        self.__buffer = numpy.append(self.__buffer, data)
        result = len(self.__buffer) >= self.__buffer_size
        # print('__buffer size %i'%self.__buffer.size)
        return result

    # Pull a portion of the buffer to process
    # (pulled samples are deleted after being
    # processed
    def get_frame(self):
        window = self.__buffer[:self.__buffer_size]
        self.__buffer = self.__buffer[self.__step:]
        # print('__buffer size %i'%self.__buffer.size)
        return window

    # Adds new audio samples to the internal
    # buffer and process them
    def process(self, data):
        if self.add_samples(data):
            while len(self.__buffer) >= self.__buffer_size:
                # Framing
                window = self.get_frame()
                # print('window size %i'%window.size)
                if self.vad(window):  # speech frame
                	self.__out_buffer = numpy.append(self.__out_buffer, window)
                # print('__out_buffer size %i'%self.__out_buffer.size)

    def get_voice_samples(self):
        return self.__out_buffer



def get_customer_voice(file):
	#print('to make it work - you need to pass 2 arguments: wav file of conversation-call and give your own filename - to create 2 files for voices')
	wav = wf.read(file)
	ch = wav[1].shape[1]
	sr = wav[0]

	c1 = wav[1][:,1]

	print('c0 %i'%c0.size)

	if ch==1:
	    exit()

	vad = VoiceActivityDetection()
	vad.process(c1)
	voice_samples = vad.get_voice_samples()
	wf.write('%s_customer.wav'%file,sr,voice_samples)


if __name__ == '__main__':
	print('working.......................')
	#print('to make it work - you need to pass 2 arguments: wav file of conversation-call and give your own filename - to create 2 files for voices')
	wav = wf.read(sys.argv[1])
	ch = wav[1].shape[1]
	sr = wav[0]

	c0 = wav[1][:,0]
	c1 = wav[1][:,1]

	print('c0 %i'%c0.size)

	vad = VoiceActivityDetection()
	vad.process(c0)
	voice_samples = vad.get_voice_samples()
	wf.write('%s_worker.wav'%sys.argv[1],sr,voice_samples)

	if ch==1:
	    exit()

	vad = VoiceActivityDetection()
	vad.process(c1)
	voice_samples = vad.get_voice_samples()
	wf.write('%s_customer.wav'%sys.argv[1],sr,voice_samples)
