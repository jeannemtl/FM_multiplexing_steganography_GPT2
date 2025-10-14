#!/usr/bin/env python3
"""
SIMPLE: Add this to the END of your GPT-2 script to create animated GIF
Just paste this entire function and call it before the final print statements
"""

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import imageio


def create_gpt2_animated_gif(token_selections, agents_config, generated_text):
    """
    Create animated GIF showing frequency signatures emerging
    
    Call this at the end of demonstrate_fm_gpt2_steganography() with:
        create_gpt2_animated_gif(token_selections, agents_config, generated_text)
    """
    print("\n" + "="*80)
    print("CREATING ANIMATED GIF")
    print("="*80)
    
    # Keyframes to show
    keyframes = list(range(20, 101, 10)) + list(range(110, 201, 10))
    keyframes.extend([200] * 8)  # Hold final frame
    
    frame_files = []
    print(f"Generating {len(keyframes)} frames...", end='', flush=True)
    
    for frame_idx, n_tokens in enumerate(keyframes):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        
        # 1. Token sequence
        tokens = token_selections[:n_tokens]
        axes[0, 0].plot(tokens, 'ko-', linewidth=1, markersize=3)
        axes[0, 0].set_title(f'Token Sequence: {n_tokens}/200 tokens', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Token Position')
        axes[0, 0].set_ylabel('Token ID')
        axes[0, 0].set_xlim(0, 200)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Frequency spectrum
        if n_tokens >= 30:
            signal = np.array(tokens, dtype=float)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            spectrum = np.abs(fft(signal))
            freqs = fftfreq(len(signal), d=1.0)
            
            pos_mask = (freqs > 0) & (freqs < 0.2)
            pos_freqs = freqs[pos_mask]
            pos_spectrum = spectrum[pos_mask]
            
            axes[0, 1].plot(pos_freqs, pos_spectrum, 'k-', linewidth=2)
            
            for agent, config in agents_config.items():
                freq = config['freq']
                color = config['color']
                axes[0, 1].axvline(freq, color=color, linestyle='--', linewidth=2.5,
                                  label=f"{agent} ({freq:.2f} Hz)", alpha=0.8)
                
                # Show power at frequency
                mask = (pos_freqs > freq-0.01) & (pos_freqs < freq+0.01)
                if np.any(mask):
                    power = np.sum(pos_spectrum[mask])
                    y_max = np.max(pos_spectrum)
                    axes[0, 1].text(freq, y_max*0.85, f'{power:.0f}',
                                   ha='center', fontsize=9, color=color, fontweight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            axes[0, 1].text(0.5, 0.5, f'Need 30+ tokens\nCurrent: {n_tokens}',
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=12, style='italic')
        
        axes[0, 1].set_title('Frequency Analysis (FFT)', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].legend(loc='upper right', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Progress
        pct = n_tokens / 200
        axes[1, 0].barh([0], [n_tokens], color='green', alpha=0.7, height=0.6)
        axes[1, 0].barh([0], [200-n_tokens], left=n_tokens, color='lightgray', 
                       alpha=0.4, height=0.6)
        axes[1, 0].text(100, 0, f'{n_tokens}/200 ({pct*100:.0f}%)',
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlim(0, 200)
        axes[1, 0].set_ylim(-1, 1)
        axes[1, 0].set_title('Generation Progress', fontweight='bold', fontsize=12)
        axes[1, 0].set_yticks([])
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Generated text
        axes[1, 1].axis('off')
        words = generated_text.split()[:n_tokens]
        recent_text = ' '.join(words[-40:]) if len(words) > 40 else ' '.join(words)
        
        # Wrap text
        lines = []
        line = []
        for word in recent_text.split():
            line.append(word)
            if len(' '.join(line)) > 45:
                lines.append(' '.join(line))
                line = []
        if line:
            lines.append(' '.join(line))
        
        wrapped = '\n'.join(lines[-6:])
        
        axes[1, 1].text(0.05, 0.95, 'Generated Stegotext:', fontweight='bold', 
                       fontsize=11, va='top')
        axes[1, 1].text(0.05, 0.82, f'"{wrapped}"', fontsize=9, va='top', 
                       style='italic', wrap=True)
        axes[1, 1].text(0.05, 0.15, '🔐 3 Hidden Channels\nFrequency-Multiplexed',
                       fontsize=10, va='top', color='red', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        fig.suptitle('FM-Multiplexed Steganography with GPT-2', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'/tmp/gpt2_gif_{frame_idx:03d}.png'
        plt.savefig(filename, dpi=85, facecolor='white')
        plt.close()
        frame_files.append(filename)
        
        if (frame_idx + 1) % 5 == 0:
            print('.', end='', flush=True)
    
    print(' Done!')
    
    # Create GIF
    print("Compiling GIF...", end='', flush=True)
    images = [imageio.imread(f) for f in frame_files]
    
    # Variable speed
    durations = [0.3] * 8 + [0.15] * (len(images) - 16) + [0.4] * 8
    
    output = 'gpt2_steganography_animated.gif'
    imageio.mimsave(output, images, duration=durations, loop=0)
    print(' Done!')
    
    # Cleanup
    for f in frame_files:
        import os
        os.remove(f)
    
    import os
    size_mb = os.path.getsize(output) / (1024*1024)
    
    print("\n" + "="*80)
    print("✅ ANIMATED GIF CREATED")
    print("="*80)
    print(f"File: {output}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"Frames: {len(images)}")
    print(f"Duration: ~{sum(durations):.1f}s")
    print("\nDownload with:")
    print(f"scp -P 18837 -i ~/.ssh/id_ed25519 root@216.81.248.113:/workspace/{output} ~/Downloads/")
    print("="*80)


# FOR TESTING: Run standalone with simulated data
if __name__ == "__main__":
    print("Creating GIF with simulated data...")
    print("(Integrate with your GPT-2 script for real results)\n")
    
    # Simulated data
    np.random.seed(42)
    base_tokens = np.random.randint(100, 15000, 200)
    t = np.arange(200)
    signal = base_tokens + 3000*np.sin(2*np.pi*0.05*t/200) + 2500*np.sin(2*np.pi*0.10*t/200) + 2000*np.sin(2*np.pi*0.15*t/200)
    token_selections = signal.astype(int).tolist()
    
    agents_config = {
        'ALICE': {'freq': 0.05, 'color': 'blue'},
        'BOB': {'freq': 0.10, 'color': 'green'},
        'CHARLIE': {'freq': 0.15, 'color': 'red'}
    }
    
    generated_text = "The future of artificial intelligence and machine learning is transforming how we interact with technology. " * 10
    
    create_gpt2_animated_gif(token_selections, agents_config, generated_text)
    print("\n✅ Test GIF created! Check gpt2_steganography_animated.gif")
