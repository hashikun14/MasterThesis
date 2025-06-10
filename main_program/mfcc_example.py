import librosa, os
import numpy as np
import matplotlib.pyplot as plt
project_root=os.path.dirname(os.getcwd())
ap=project_root+"\example\Lucio Dalla.mp3"

y, sr = librosa.load(ap,sr=None,mono=True, offset=15, duration=60)

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Waveform.png") 
librosa.display.waveshow(y, sr=sr, color='blue')
plt.title("Waveform of L'anno che verrà", fontsize=12)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig(os.path.join(file_path))

mfcc_dalla=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,hop_length=256,n_fft=1024,n_mels=40)

duration = len(y)/sr
time_points = np.linspace(0, duration, num=mfcc_dalla.shape[1])
file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MFCC Example.png") 
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_dalla, x_coords=time_points, x_axis='time', y_axis='mel',hop_length=256, sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC Spectrogram')
plt.ylabel('Hertz')
plt.tight_layout()
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Low MFCCs.png") 
plt.figure(figsize=(13, 6))
plt.plot(mfcc_dalla[0], label='MFCC 1')
plt.plot(mfcc_dalla[1], label='MFCC 2')
plt.plot(mfcc_dalla[2], label='MFCC 3')
plt.title("Low MFCCs: Vocal Tract Features")
plt.legend(loc='center left', bbox_to_anchor=(1.0005, 0.5))
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","High MFCCs.png") 
plt.figure(figsize=(13, 6))
plt.plot(mfcc_dalla[-3], label='MFCC 11')
plt.plot(mfcc_dalla[-2], label='MFCC 12')
plt.plot(mfcc_dalla[-1], label='MFCC 13')
plt.title("High MFCCs: Excitation Features")
plt.legend(loc='center left', bbox_to_anchor=(1.0005, 0.5))
plt.savefig(os.path.join(file_path))