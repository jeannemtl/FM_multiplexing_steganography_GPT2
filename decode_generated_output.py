import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class FrequencyMultiplexDecoder:
    """
    Decoder for frequency-multiplexed steganography in GPT-2 generated text.
    """
    
    def __init__(self, model_name='gpt2'):
        """Initialize decoder with GPT-2 model."""
        print("Loading GPT-2 model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Agent configuration - YOUR FREQUENCIES
        self.agents = {
            'ALICE': {'freq': 0.02, 'color': 'blue', 'original_bits': np.array([0, 1, 0, 0, 0, 1, 0, 0])},
            'BOB': {'freq': 0.04, 'color': 'green', 'original_bits': np.array([1, 1, 1, 0, 1, 0, 1, 1])},
            'CHARLIE': {'freq': 0.06, 'color': 'red', 'original_bits': np.array([1, 1, 1, 0, 1, 0, 0, 0])}
        }
        
    def extract_token_entropy(self, stegotext, context="The future of artificial intelligence"):
        """
        Extract token-by-token entropy sequence from stegotext.
        """
        # Tokenize
        full_text = context + " " + stegotext
        input_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
        context_length = len(self.tokenizer.encode(context))
        
        entropy_sequence = []
        tokens = []
        
        print(f"\nExtracting entropy from {len(input_ids[0]) - context_length} tokens...")
        
        with torch.no_grad():
            for i in range(context_length, len(input_ids[0])):
                # Get model predictions at this position
                outputs = self.model(input_ids[:, :i])
                logits = outputs.logits[0, -1, :]  # Last token predictions
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=0)
                
                # Calculate entropy: H = -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                entropy_sequence.append(entropy)
                
                # Store token
                token = self.tokenizer.decode([input_ids[0, i].item()])
                tokens.append(token)
                
                if (i - context_length + 1) % 100 == 0:
                    print(f"  Processed {i - context_length + 1} tokens...")
        
        print(f"✓ Extracted {len(entropy_sequence)} entropy values")
        return np.array(entropy_sequence), tokens
    
    def fft_analysis(self, signal):
        """Perform FFT and detect peaks at known frequencies."""
        N = len(signal)
        normalized = (signal - np.mean(signal)) / np.std(signal)
        
        # Compute FFT
        fft_vals = fft(normalized)
        fft_freq = fftfreq(N, d=1.0)
        
        # Positive frequencies only
        pos_mask = fft_freq > 0
        freqs = fft_freq[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        
        print(f"\n{'='*80}")
        print(f"FFT ANALYSIS ({N} samples)")
        print(f"{'='*80}")
        print(f"Frequency resolution: {1/N:.4f} Hz\n")
        
        detected_powers = {}
        
        # Check for peaks at expected frequencies
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            
            # Integrate power in window around target (coherent integration)
            window_size = 30  # ~30 frequency bins as mentioned in paper
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            start_idx = max(0, freq_idx - window_size//2)
            end_idx = min(len(freqs), freq_idx + window_size//2)
            
            integrated_power = np.sum(power[start_idx:end_idx])
            detected_freq = freqs[freq_idx]
            
            detected_powers[agent_name] = integrated_power
            
            print(f"{agent_name:8s} @ {target_freq:.3f} Hz:")
            print(f"  Detected at: {detected_freq:.4f} Hz")
            print(f"  Integrated power: {integrated_power:.1f} units")
            print(f"  Detection error: {abs(detected_freq - target_freq):.4f} Hz")
            print()
        
        # Calculate average noise level (exclude signal regions)
        noise_mask = np.ones(len(freqs), dtype=bool)
        for agent_name, agent_info in self.agents.items():
            target_freq = agent_info['freq']
            noise_mask &= (np.abs(freqs - target_freq) > 0.02)  # Exclude ±0.02 Hz around signals
        
        avg_noise = np.mean(power[noise_mask]) if np.any(noise_mask) else np.mean(power)
        
        print(f"Average noise level: {avg_noise:.1f} units")
        print(f"Signal-to-noise ratios:")
        for agent_name, integrated_power in detected_powers.items():
            snr = integrated_power / (avg_noise * 30)  # Normalize by window size
            print(f"  {agent_name:8s}: {snr:.2f}:1")
        
        return freqs, power, detected_powers
    
    def bandpass_filter(self, signal, center_freq, bandwidth=0.015):
        """Apply bandpass filter around target frequency."""
        fs = 1.0  # 1 sample per token
        nyquist = fs / 2
        low = (center_freq - bandwidth/2) / nyquist
        high = (center_freq + bandwidth/2) / nyquist
        
        # Ensure valid frequency range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        # 4th order Butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        return filtered
    
    def extract_amplitude_envelope(self, filtered_signal):
        """Extract amplitude envelope using Hilbert transform."""
        # Analytic signal via FFT
        fft_signal = np.fft.fft(filtered_signal)
        n = len(fft_signal)
        
        # Zero out negative frequencies (Hilbert transform)
        fft_signal[n//2 + 1:] = 0
        fft_signal[1:n//2] *= 2
        
        # Back to time domain
        analytic_signal = np.fft.ifft(fft_signal)
        envelope = np.abs(analytic_signal)
        
        return envelope
    
    def decode_bits_kmeans(self, envelope, n_bits=16):
        """Decode bits using k-means clustering."""
        from sklearn.cluster import KMeans
        
        # Resample envelope to match expected bit length
        if len(envelope) > n_bits:
            # Average over windows
            bits_per_window = len(envelope) // n_bits
            resampled = np.array([
                np.mean(envelope[i*bits_per_window:(i+1)*bits_per_window])
                for i in range(n_bits)
            ])
        else:
            resampled = envelope[:n_bits]
        
        # K-means clustering with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(resampled.reshape(-1, 1))
        
        # Assign 0/1 based on cluster centers
        centers = kmeans.cluster_centers_.flatten()
        if centers[0] > centers[1]:
            bits = 1 - labels  # Higher cluster = 1
        else:
            bits = labels
        
        # Calculate confidence
        boundary = np.mean(centers)
        confidence = np.abs(resampled - boundary)
        avg_confidence = np.mean(confidence)
        
        return bits, avg_confidence
    
    def decode_bits_threshold(self, envelope, n_bits=16):
        """Decode bits using median threshold."""
        # Resample
        if len(envelope) > n_bits:
            bits_per_window = len(envelope) // n_bits
            resampled = np.array([
                np.mean(envelope[i*bits_per_window:(i+1)*bits_per_window])
                for i in range(n_bits)
            ])
        else:
            resampled = envelope[:n_bits]
        
        # Threshold at median
        threshold = np.median(resampled)
        bits = (resampled > threshold).astype(int)
        confidence = np.abs(resampled - threshold)
        
        return bits, np.mean(confidence)
    
    def decode_agent(self, entropy_sequence, agent_name, method='kmeans', n_bits=16):
        """
        Decode one agent's hidden message.
        """
        agent = self.agents[agent_name]
        freq = agent['freq']
        
        print(f"\n{'='*80}")
        print(f"DECODING {agent_name} @ {freq} Hz")
        print(f"{'='*80}")
        
        # Step 1: Normalize
        normalized = (entropy_sequence - np.mean(entropy_sequence)) / np.std(entropy_sequence)
        print(f"✓ Normalized entropy sequence (μ=0, σ=1)")
        
        # Step 2: Bandpass filter
        print(f"✓ Applying bandpass filter (bandwidth=0.015 Hz)...")
        filtered = self.bandpass_filter(normalized, freq)
        
        # Step 3: Extract envelope
        print(f"✓ Extracting amplitude envelope (Hilbert transform)...")
        envelope = self.extract_amplitude_envelope(filtered)
        
        # Step 4: Decode bits
        print(f"✓ Decoding bits using {method} method...")
        if method == 'kmeans':
            bits, confidence = self.decode_bits_kmeans(envelope, n_bits)
        else:
            bits, confidence = self.decode_bits_threshold(envelope, n_bits)
        
        # Compare to original
        original_bits = agent['original_bits'][:n_bits]
        if len(bits) >= len(original_bits):
            matches = np.sum(bits[:len(original_bits)] == original_bits)
            accuracy = matches / len(original_bits) * 100
            ber = 100 - accuracy
        else:
            matches = np.sum(bits == original_bits[:len(bits)])
            accuracy = matches / len(bits) * 100
            ber = 100 - accuracy
        
        print(f"\nRESULTS:")
        print(f"  Decoded bits:  {self._format_bits(bits)}")
        print(f"  Original bits: {self._format_bits(original_bits)}")
        print(f"  Matches: {matches}/{min(len(bits), len(original_bits))}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Bit Error Rate: {ber:.1f}%")
        print(f"  Confidence: {confidence:.3f}")
        
        return bits, accuracy, ber, confidence, filtered, envelope
    
    def decode_all(self, stegotext, context="The future of artificial intelligence", 
                   method='kmeans', n_bits=16):
        """
        Complete decoding pipeline for all three agents.
        """
        print("\n" + "="*80)
        print("FREQUENCY-DIVISION MULTIPLEXED STEGANOGRAPHY DECODER")
        print("="*80)
        
        # Step 1: Extract entropy
        print("\n[STEP 1] EXTRACTING TOKEN ENTROPY SEQUENCE")
        print("-"*80)
        entropy_seq, tokens = self.extract_token_entropy(stegotext, context)
        
        # Step 2: FFT Analysis
        print("\n[STEP 2] FFT ANALYSIS & CARRIER DETECTION")
        print("-"*80)
        freqs, power, detected_powers = self.fft_analysis(entropy_seq)
        
        # Step 3: Decode each agent
        print("\n[STEP 3] INDIVIDUAL AGENT DECODING")
        print("-"*80)
        
        results = {}
        all_filtered = {}
        all_envelopes = {}
        
        for agent_name in ['ALICE', 'BOB', 'CHARLIE']:
            bits, acc, ber, conf, filtered, envelope = self.decode_agent(
                entropy_seq, agent_name, method=method, n_bits=n_bits
            )
            
            results[agent_name] = {
                'bits': bits,
                'accuracy': acc,
                'ber': ber,
                'confidence': conf,
                'frequency': self.agents[agent_name]['freq'],
                'power': detected_powers[agent_name]
            }
            
            all_filtered[agent_name] = filtered
            all_envelopes[agent_name] = envelope
        
        # Step 4: Visualizations
        print("\n[STEP 4] GENERATING VISUALIZATIONS")
        print("-"*80)
        self._plot_fft_spectrum(freqs, power)
        self._plot_decoding_pipeline(entropy_seq, all_filtered, all_envelopes, results)
        
        # Summary
        self._print_summary(results)
        
        return results, entropy_seq, tokens
    
    def _plot_fft_spectrum(self, freqs, power):
        """Plot FFT spectrum showing all three carrier frequencies."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot spectrum
        ax.plot(freqs, power, 'k-', linewidth=1, alpha=0.7)
        
        # Mark carrier frequencies
        for agent_name, agent_info in self.agents.items():
            freq = agent_info['freq']
            color = agent_info['color']
            
            # Find peak
            idx = np.argmin(np.abs(freqs - freq))
            ax.axvline(freq, color=color, linestyle='--', linewidth=2, 
                      label=f'{agent_name} ({freq} Hz)', alpha=0.7)
            ax.plot(freqs[idx], power[idx], 'o', color=color, markersize=10)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power (units²)', fontsize=12)
        ax.set_title('FFT Power Spectrum - Frequency-Multiplexed Carriers', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 0.15])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fft_spectrum.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: fft_spectrum.png")
        plt.close()
    
    def _plot_decoding_pipeline(self, entropy, filtered_signals, envelopes, results):
        """Plot complete decoding pipeline for all agents."""
        fig, axes = plt.subplots(4, 3, figsize=(18, 12))
        
        time = np.arange(len(entropy))
        
        for col, agent_name in enumerate(['ALICE', 'BOB', 'CHARLIE']):
            color = self.agents[agent_name]['color']
            freq = self.agents[agent_name]['freq']
            filtered = filtered_signals[agent_name]
            envelope = envelopes[agent_name]
            bits = results[agent_name]['bits']
            ber = results[agent_name]['ber']
            
            # Row 0: Original entropy
            if col == 0:
                axes[0, col].set_ylabel('Entropy\n(bits)', fontsize=10)
            axes[0, col].plot(time, entropy, 'k-', alpha=0.4, linewidth=0.5)
            axes[0, col].set_title(f'{agent_name}\n{freq} Hz', fontsize=12, fontweight='bold')
            axes[0, col].grid(True, alpha=0.3)
            axes[0, col].set_xlim([0, len(entropy)])
            
            # Row 1: Bandpass filtered
            if col == 0:
                axes[1, col].set_ylabel('Filtered\nAmplitude', fontsize=10)
            axes[1, col].plot(time, filtered, color=color, linewidth=0.8)
            axes[1, col].grid(True, alpha=0.3)
            axes[1, col].set_xlim([0, len(entropy)])
            
            # Row 2: Envelope
            if col == 0:
                axes[2, col].set_ylabel('Envelope\nAmplitude', fontsize=10)
            axes[2, col].plot(time, envelope, color=color, linewidth=1.2)
            axes[2, col].grid(True, alpha=0.3)
            axes[2, col].set_xlim([0, len(entropy)])
            
            # Row 3: Decoded bits
            if col == 0:
                axes[3, col].set_ylabel('Decoded\nBits', fontsize=10)
            bit_time = np.linspace(0, len(entropy), len(bits))
            axes[3, col].step(bit_time, bits, color=color, linewidth=2, where='mid')
            axes[3, col].set_ylim([-0.1, 1.1])
            axes[3, col].set_xlabel('Token Position', fontsize=10)
            axes[3, col].grid(True, alpha=0.3)
            axes[3, col].set_xlim([0, len(entropy)])
            axes[3, col].text(0.98, 0.95, f'BER: {ber:.1f}%', 
                            transform=axes[3, col].transAxes,
                            fontsize=10, ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Row labels
        row_labels = ['Original\nEntropy', 'Bandpass\nFiltered', 'Amplitude\nEnvelope', 'Decoded\nBits']
        for row in range(4):
            axes[row, 0].text(-0.3, 0.5, row_labels[row], 
                            transform=axes[row, 0].transAxes,
                            fontsize=11, ha='center', va='center', 
                            rotation=90, fontweight='bold')
        
        plt.suptitle('Decoding Pipeline: Frequency-Division Multiplexed Steganography', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('decoding_pipeline.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: decoding_pipeline.png")
        plt.close()
    
    def _print_summary(self, results):
        """Print final summary table."""
        print("\n" + "="*80)
        print("DECODING SUMMARY")
        print("="*80)
        print(f"{'Agent':<10} {'Freq (Hz)':<12} {'Power':<12} {'Accuracy':<12} {'BER':<12} {'Confidence':<12}")
        print("-"*80)
        
        for agent_name in ['ALICE', 'BOB', 'CHARLIE']:
            r = results[agent_name]
            print(f"{agent_name:<10} {r['frequency']:<12.3f} {r['power']:<12.1f} "
                  f"{r['accuracy']:<12.1f}% {r['ber']:<12.1f}% {r['confidence']:<12.3f}")
        
        avg_ber = np.mean([results[a]['ber'] for a in ['ALICE', 'BOB', 'CHARLIE']])
        print("-"*80)
        print(f"Average BER: {avg_ber:.1f}%")
        print("="*80)
    
    def _format_bits(self, bits):
        """Format bits as string."""
        return '[' + ' '.join(str(int(b)) for b in bits) + ']'


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Your actual generated stegotext
    STEGOTEXT = """in China is bleak, but many believe that's unlikely as China's computer and telecom sectors are thriving. According to one expert, machine intelligence technologies currently on an upward trajectory are unlikely to reach all the major US cities that have opened since the 1990s, unless their technology can be quickly used to solve the country's long and complex problems, a strategy many have deemed obsolete.


In recent weeks, Chinese telecom companies have begun to launch major networks at scale, opening the first national Internet service provider, allowing access to large-scale Internet service around the world, as if at a huge scale.


However, there remain many concerns that state-controlled technology could run into major opposition.


In recent weeks, government officials have said they will begin deploying virtual reality (VR) hardware to monitor residents, with the aim of expanding virtual reality into every location connected to their homes. Although they have not released any official data about such devices, analysts say their main potential application will be for surveillance of tourists and law enforcement in remote places, such as Beijing.


B.W. Chang , professor, National Central University of Taiwan


Meanwhile, experts say they'll be careful of China's increasing influence over U.S.-based news.


Citing an unnamed person, Chinese news outlets say they will target "the first day of every month" for content by content specialists such as a "New York Times.org.cn" and "KovparuTimes.com" or "CNNChina.com."


And Chinese news websites say they will use bots and other bots, like on the "New York Times."info."com.cn or, "censor.cn," to target government officials.<|endoftext|>The European Commission has imposed penalties in line with EU law that allow telecommunications companies to challenge claims filed with the EC or the country's consumer protection chief, the Luxembourg-based law centre, known as LTRA.


The fines are aimed at limiting the scope of access that telecom companies get to monitor customer data, privacy and consumer affairs services when trying to sue in a case challenging the constitutionality of the EU's digital privacy law to stop data security breaches for the United States-based, European carrier Carrier Solutions, in its complaint to European Commission President Herman Van Rompuy.


Under the previous Irish regime, telecommunications firms must pay about a 40-year penalty after four years of the levy depending on whether they actually win in the court of public opinion on both damages and appeals – including a 50- to 80-euro sum for the losses caused.


Under the proposed measures, the maximum penalty can be up to five years.


"In line with our agreement for the initial two years of 2013, the commission has imposed a fine of €9,000 that applies on the basis of the amount of data that has been accessed or acquired by third parties," European commission spokesman Martin Kaslin said.


Under the EU privacy law, an electronic service company must provide access to customer data in the event of a national court decision requiring the service to protect customers' privacy.


A case claiming infringement of the EU's privacy law will be prosecuted by the Commission from the outset regardless of its merits, with either side expected to decide in a year's time whether the company should be penalised.


The country's former telecommunications commissioner Euïnio Guzman took the stance publicly in March 2016 when a group of EU telecoms associations protested the new national data data protection law in Luxembourg by pointing out its use of such technologies as telephony data, for instance, can damage consumers' privacy rights.


During the summer of this year, Telecom Minister Tafisa Asori responded to the protests publicly in an attempt to counter claims by AT&T by asking why she wanted to scrap the national data protection laws in the interest of consumers as such measures may lead to more customers being targeted by AT&T.


The minister's comments triggered an explosion of anti-data abuse campaigns outside the Commission headquarters demanding an explanation from its chairman Frans Timmermans and the chairperson of the European Parliament's Information Commission Nils Nohl.


Following this, the Commission imposed a €200 fine against AT&T's president Christian Prinsen, according to the news agency Rada. An anti-data abuse law is considered to be the "prolonged right" used in EU law to collect personal data, if two-thirds of the law's judges agree with its proponents.


Although both parties agree the EU uses a national data protection law for its surveillance of telecommunications users, telecoms operators such as telecommunications giants such as AT&T and Bell are barred from using this protection as part of data sharing legislation.


The bill was published on Thursday. However, with little official detail, the bill says consumers can take complaints to the telecommunications regulator of their content preferences"""

    CONTEXT = "The future of artificial intelligence"
    
    # Initialize decoder
    decoder = FrequencyMultiplexDecoder()
    
    # Run complete decoding
    results, entropy, tokens = decoder.decode_all(
        STEGOTEXT,
        context=CONTEXT,
        method='kmeans',  # or 'threshold'
        n_bits=16  # Match your 16-bit messages (you only used first 8 bits)
    )
    
    print("\n✓ Decoding complete!")
    print("\nGenerated files:")
    print("  - fft_spectrum.png")
    print("  - decoding_pipeline.png")
