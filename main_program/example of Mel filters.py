import numpy as np
import matplotlib.pyplot as plt
import os
project_root=os.path.dirname(os.getcwd())
file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","example of Mel filters.png") 
def hz_to_mel(f):
    return 2595*np.log10(1+f/700)

def mel_to_hz(m):
    return 700*(10**(m/2595)-1)

n_filters=10
n_fft=512
sample_rate=22050
min_freq=0
max_freq=sample_rate//2

min_mel=hz_to_mel(min_freq)
max_mel=hz_to_mel(max_freq)
mel_points=np.linspace(min_mel, max_mel, n_filters+2)
hz_points=mel_to_hz(mel_points)

bins=np.floor((n_fft+1)*hz_points/sample_rate).astype(int)

plt.figure(figsize=(10, 6))
plt.title('Triangular Filters in Mel Filterbank')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Weight')

k=5
highlight_indices=[k-1, k, k+1]
colors=['blue', 'red', 'green']
labels=[f'Filter {k-1} (f$_{{k-1}}$)', f'Filter {k} (f$_k$)', f'Filter {k+1} (f$_{{k+1}}$)']

for i in range(n_filters):
    start, center, end=hz_points[i:i+3]
    freq=np.linspace(start, end, 500)
    filter_weights=np.zeros_like(freq)
    
    mask=(freq >= start) & (freq<=center)
    filter_weights[mask]=(freq[mask]-start)/(center-start)
    
    mask=(freq >= center) & (freq<=end)
    filter_weights[mask]=(end-freq[mask])/(end-center)
    
    if i in highlight_indices:
        idx=highlight_indices.index(i)
        plt.plot(freq, filter_weights, color=colors[idx], linewidth=2, label=labels[idx])
        
        if i == k:
            plt.text(center, 1.05, f'f$_k$={center:.0f} Hz', ha='center')
        elif i == k-1:
            plt.text(center, 1.05, f'f$_{{k-1}}$={center:.0f} Hz', ha='center')
        elif i == k+1:
            plt.text(center, 1.05, f'f$_{{k+1}}$={center:.0f} Hz', ha='center')
    else:
        plt.plot(freq, filter_weights, 'k-', alpha=0.2)

plt.legend()
plt.grid(True)
plt.ylim(0, 1.2)
plt.tight_layout()
plt.savefig(os.path.join(file_path))
