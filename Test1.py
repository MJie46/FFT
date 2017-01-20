# -*- coding: UTF-8 -*-

import pymatlab
import numpy
import wave
import pylab as pl
from scipy import signal

# 打开wav文件
# open返回一个的是一个Wave_read类的实例，通过调用它的方法读取WAV文件的格式和数据
f = wave.open("4.wav", "rb")

# 读取格式信息
# 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）, 采
# 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
print nchannels,sampwidth,framerate,nframes

# 读取波形数据
# 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
str_data = f.readframes(nframes)
f.close()

# 将波形数据转换成数组
# 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
wave_data1 = numpy.fromstring(str_data, dtype=numpy.short)
wave_data = wave_data1.copy()
wave_data.shape = -1, 2
wave_data = wave_data.T
time = numpy.arange(0, nframes) * (1.0 / framerate)
len_time = len(time) / 2
#time = time[0:len_time]

##print "time length = ",len(time)
##print "wave_data[0] length = ",len(wave_data[0])

# pl.subplot(211)
# pl.plot(time, wave_data[0])
# pl.subplot(212)
# pl.plot(time, wave_data[1], c="r")
# pl.show()

# 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
N=nframes
start=0 #开始采样位置
df = framerate/(N-1) # 分辨率
freq = [df*n for n in range(0,N)] #N个元素
wave_data2=wave_data[0][start:start+N]
wave_data3=wave_data[1][start:start+N]



c=numpy.fft.fft(wave_data2)
c_shift=numpy.fft.fftshift(c)
# c_shift = c_shift*12.5
c_mask = numpy.fft.ifft(numpy.fft.ifftshift(c_shift))
lpfImg = numpy.real(c_mask)

c1=numpy.fft.fft(wave_data3)
c_shift1=numpy.fft.fftshift(c1)
# c_shift1 = c_shift1*12.5
c_mask1 = numpy.fft.ifft(numpy.fft.ifftshift(c_shift1))
lpfImg1 = numpy.real(c_mask1)


n = len(lpfImg)+len(lpfImg1)
wave45 = numpy.zeros(n, numpy.int16)
wave45[0:n:2] = lpfImg[:]
wave45[1:n:2] = lpfImg1[:]

# 打开WAV文档
f = wave.open(r"45.wav", "wb")
# 配置声道数、量化位数和取样频率
f.setnchannels(nchannels)
f.setsampwidth(sampwidth)
f.setframerate(framerate)
f.setnframes(nframes)
# 将wav_data转换为二进制数据写入文件
f.writeframes(wave45.tostring())
f.close()

pl.subplot(411)
pl.plot(wave_data1,'r')
pl.subplot(412)
pl.plot(wave_data2,'g')
pl.subplot(413)
pl.plot(wave_data3,'g')
pl.subplot(414)
pl.plot(wave45,'b')
pl.show()

c1=c
c2 = numpy.fft.ifft(c)
#常规显示采样频率一半的频谱
d=int(len(c)/2)
#仅显示频率在4000以下的频谱
while freq[d]>4000:
    d-=10

pl.subplot(211)
pl.plot(freq[:d-1],abs(c[:d-1]),'r')
pl.subplot(212)
pl.plot(freq[:d-1],abs(c1[:d-1]),'b')
pl.show()












# framerate = 44100
# time = 10
# # 产生10秒44.1kHz的100Hz - 1kHz的频率扫描波
# t = numpy.arange(0, time, 1.0/framerate)
# wave_data = signal.chirp(t, 100, time, 1000, method='linear') * 5000
# wave_data = wave_data.astype(numpy.short)
# # 打开WAV文档
# f = wave.open(r"44.wav", "wb")
# # 配置声道数、量化位数和取样频率
# f.setnchannels(nchannels)
# f.setsampwidth(sampwidth)
# f.setframerate(framerate)
# # 将wav_data转换为二进制数据写入文件
# f.writeframes(str_data.tostring())
# f.close()

# from gpcharts import figure
# my_plot = figure(title='test')
# my_plot.plot([1,2,10,20,50,100])

# import matplotlib.pyplot as plt
# t = numpy.arange(400)
# n = numpy.zeros((400,), dtype=complex)
# n[40:60] = numpy.exp(1j*numpy.random.uniform(0, 2*numpy.pi, (20,)))
# s = numpy.fft.ifft(n)
# pl.subplot(211)
# pl.plot(t, s.real, 'b-')
# pl.subplot(212)
# pl.plot(n)
# pl.legend(('real', 'imaginary'))
# pl.show()