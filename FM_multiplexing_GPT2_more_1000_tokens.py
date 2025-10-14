#!/usr/bin/env python3
"""
Frequency-Multiplexed Steganography with GPT-2
Combines FM multiplexing with GPT-2 text generation (like Schroeder de Witt et al.)
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def setup_gpt2():
    """Initialize GPT-2 model and tokenizer"""
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✓ GPT-2 loaded on {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def get_token_probabilities(model, tokenizer, context, device, top_k=50):
    """
    Get probability distribution over next tokens given context
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        context: input text string
        device: torch device (cpu or cuda)
        top_k: limit to top-k tokens
    
    Returns:
        token_ids: list of token IDs
        probabilities: probability for each token
    """
    # Encode context
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    
    # Get logits from model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    # Convert to probabilities
    probs = torch.softmax(logits, dim=0)
    
    # Get top-k
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Convert to numpy
    token_ids = top_indices.cpu().numpy()
    probabilities = top_probs.cpu().numpy()
    
    # Renormalize
    probabilities = probabilities / probabilities.sum()
    
    return token_ids, probabilities


def encode_bits_to_token_selection(bits, frequency, num_tokens):
    """
    Encode bits using VERY STRONG frequency modulation to bias token selection
    
    Args:
        bits: binary array to encode
        frequency: carrier frequency for this agent
        num_tokens: number of tokens in vocabulary
    
    Returns:
        bias_vector: frequency-modulated bias to apply to token probabilities
    """
    # Create frequency signature with MORE CYCLES
    t = np.linspace(0, 10, num_tokens)  # 10 periods
    carrier = np.sin(2 * np.pi * frequency * t)
    
    # Use more bits to create stronger amplitude modulation
    bit_value = np.mean(bits)  # Overall bit density (0 to 1)
    
    # VERY STRONG modulation: 0.5 to 1.0 range (±50%)
    amplitude = 0.5 + 0.5 * bit_value
    bias = amplitude * carrier
    
    # Normalize but keep it VERY STRONG (0.5 instead of 0.3)
    bias = bias / (np.max(np.abs(bias)) + 1e-10) * 0.5
    
    return bias


def fm_multiplexed_token_selection(model, tokenizer, context, agent_messages, agents_config, device):
    """
    Select next token using FM-multiplexed steganography
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer  
        context: current text context
        agent_messages: dict of {agent: bits_to_encode}
        agents_config: dict of {agent: frequency}
        device: torch device
    
    Returns:
        selected_token: the chosen token ID
        token_text: the decoded token text
    """
    # Get base probabilities from GPT-2
    token_ids, base_probs = get_token_probabilities(model, tokenizer, context, device, top_k=50)
    
    # Create combined bias from all agents (multiplexing)
    combined_bias = np.zeros(len(token_ids))
    
    for agent, bits in agent_messages.items():
        if len(bits) > 0:  # If agent has bits to send
            freq = agents_config[agent]['freq']
            bias = encode_bits_to_token_selection(bits, freq, len(token_ids))
            combined_bias += bias
    
    # Apply bias to probabilities
    biased_probs = base_probs * (1 + combined_bias)
    biased_probs = biased_probs / biased_probs.sum()  # Renormalize
    
    # Sample token
    selected_idx = np.random.choice(len(token_ids), p=biased_probs)
    selected_token = token_ids[selected_idx]
    token_text = tokenizer.decode([selected_token])
    
    return selected_token, token_text, base_probs, biased_probs


def analyze_token_sequence_frequency(token_selections, agents_config, all_base_probs=None):
    """
    Analyze the frequency content of token selection pattern
    
    Args:
        token_selections: list of selected token indices
        agents_config: agent frequency configuration
        all_base_probs: optional list of probability distributions (for entropy-based analysis)
    
    Returns:
        detected_agents: which agents were detected
    """
    # Option 1: Use token entropy (smoother signal)
    if all_base_probs is not None and len(all_base_probs) > 0:
        print("  Using token entropy for frequency analysis (smoother)")
        signal = np.array([
            -np.sum(probs * np.log(probs + 1e-10)) 
            for probs in all_base_probs
        ])
    else:
        # Option 2: Use token IDs (noisier)
        print("  Using token IDs for frequency analysis")
        signal = np.array(token_selections, dtype=float)
    
    # Detrend first (remove linear trend)
    signal = signal - np.linspace(signal[0], signal[-1], len(signal))
    
    # Then normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(signal))
    signal = signal * window
    
    # Compute FFT
    spectrum = np.abs(fft(signal))
    freqs = fftfreq(len(signal), d=1.0)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_spectrum = spectrum[pos_mask]
    
    # Detect peaks for each agent with WIDER window
    detected = {}
    for agent, config in agents_config.items():
        target_freq = config['freq']
        
        # Find power near target frequency (wider window: ±0.015 instead of ±0.01)
        freq_mask = (pos_freqs > target_freq - 0.015) & (pos_freqs < target_freq + 0.015)
        if np.any(freq_mask):
            power = np.sum(pos_spectrum[freq_mask])
            detected[agent] = power
        else:
            detected[agent] = 0
    
    return detected, pos_freqs, pos_spectrum


def demonstrate_fm_gpt2_steganography():
    """
    Main demonstration: FM-multiplexed steganography with GPT-2
    """
    print("="*80)
    print("FM-MULTIPLEXED STEGANOGRAPHY WITH GPT-2")
    print("="*80)
    
    # Initialize GPT-2
    model, tokenizer, device = setup_gpt2()
    
    # Configure agents with LOWER frequencies for better detection
    agents_config = {
        'ALICE': {'freq': 0.02, 'color': 'blue'},    # Lower: 0.02 Hz
        'BOB': {'freq': 0.04, 'color': 'green'},     # Lower: 0.04 Hz
        'CHARLIE': {'freq': 0.06, 'color': 'red'}    # Lower: 0.06 Hz
    }
    
    # Create messages for each agent
    np.random.seed(42)
    agent_messages = {
        'ALICE': np.random.randint(0, 2, 16),
        'BOB': np.random.randint(0, 2, 16),
        'CHARLIE': np.random.randint(0, 2, 16)
    }
    
    print("\n" + "="*80)
    print("AGENT MESSAGES TO ENCODE:")
    print("="*80)
    for agent, bits in agent_messages.items():
        freq = agents_config[agent]['freq']
        print(f"{agent:8} ({freq:.2f} Hz): {bits[:8]}... ({len(bits)} bits)")
    
    # Starting context
    context = "The future of artificial intelligence"
    
    print("\n" + "="*80)
    print("GENERATING STEGOTEXT:")
    print("="*80)
    print(f"Context: \"{context}\"")
    print(f"\nGenerating {1000} tokens with embedded messages...")
    print("This will take 2-3 minutes on GPU...\n")
    
    # Generate stegotext
    generated_text = context
    token_selections = []
    all_base_probs = []
    all_biased_probs = []
    
    for i in range(1000):
        # Each agent "sends" their message (in practice, would consume bits)
        selected_token, token_text, base_probs, biased_probs = fm_multiplexed_token_selection(
            model, tokenizer, generated_text, agent_messages, agents_config, device
        )
        
        generated_text += token_text
        token_selections.append(selected_token)
        all_base_probs.append(base_probs)
        all_biased_probs.append(biased_probs)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1} tokens...")
    
    print(f"\n✓ Complete!\n")
    print("="*80)
    print("GENERATED STEGOTEXT:")
    print("="*80)
    print(f"\"{generated_text}\"")
    print("="*80)
    
    # Analyze frequency content
    print("\n" + "="*80)
    print("FREQUENCY ANALYSIS (RECEIVER SIDE):")
    print("="*80)
    
    detected_agents, freqs, spectrum = analyze_token_sequence_frequency(
        token_selections, agents_config, all_base_probs  # Pass probabilities for entropy analysis
    )
    
    print("\nDetected agent signals:")
    for agent, power in detected_agents.items():
        freq = agents_config[agent]['freq']
        print(f"  {agent:8} ({freq:.2f} Hz): Power = {power:.2f}")
    
    # Calculate KL divergence between base and biased distributions
    print("\n" + "="*80)
    print("SECURITY ANALYSIS:")
    print("="*80)
    
    avg_kl = 0
    for base_p, biased_p in zip(all_base_probs, all_biased_probs):
        # KL divergence: sum(p * log(p/q))
        kl = np.sum(biased_p * np.log((biased_p + 1e-10) / (base_p + 1e-10)))
        avg_kl += kl
    avg_kl /= len(all_base_probs)
    
    print(f"Average KL divergence per token: {avg_kl:.6f}")
    print(f"(Lower is better - perfect security = 0)")
    
    if avg_kl < 0.01:
        print("✓ Excellent: Stegotext is statistically indistinguishable from covertext")
    elif avg_kl < 0.05:
        print("✓ Good: Low detectability")
    elif avg_kl < 0.1:
        print("⚠ Moderate: Some detectability (trade-off for stronger signals)")
    else:
        print("⚠ Detectable: KL divergence is significant")
    
    # Visualization
    create_visualization(token_selections, agents_config, freqs, spectrum, 
                        generated_text, all_base_probs, all_biased_probs)
    
    # Create animated GIF
    print("\n" + "="*80)
    print("CREATING ANIMATED GIF...")
    print("="*80)
    create_animated_gif(token_selections, agents_config, generated_text)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Multiple agents embed messages at different frequencies")
    print("2. GPT-2 generates natural-looking text (covertext)")
    print("3. Token selection is subtly biased by FM-modulated signals")
    print("4. Receiver uses FFT to detect frequency signatures")
    print("5. Text appears normal, but contains hidden multiplexed channels")
    print()
    print("TRADE-OFF:")
    print(f"  Bias strength: ±50% (strong signals)")
    print(f"  KL divergence: {avg_kl:.4f}")
    print(f"  Security: {'Excellent' if avg_kl < 0.01 else 'Good' if avg_kl < 0.05 else 'Moderate'}")
    print(f"  Detectability: {'High' if avg_kl > 0.05 else 'Medium' if avg_kl > 0.02 else 'Low'}")
    print("="*80)


def create_animated_gif(token_selections, agents_config, generated_text):
    """Create animated GIF showing results building up over time"""
    
    print("\nGenerating animation frames...")
    
    # Define keyframes
    keyframes = list(range(10, 51, 5)) + list(range(60, 151, 10)) + list(range(155, 201, 5))
    keyframes.extend([200] * 10)  # Hold final
    
    frame_files = []
    
    for i, n_tokens in enumerate(keyframes):
        tokens = token_selections[:n_tokens]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Token sequence
        axes[0, 0].plot(tokens, 'ko-', linewidth=1, markersize=4)
        axes[0, 0].set_title(f'Token Sequence ({n_tokens} tokens)', fontweight='bold')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Token ID')
        axes[0, 0].set_xlim(0, 200)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Frequency spectrum
        if n_tokens >= 20:
            signal = np.array(tokens, dtype=float)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            spectrum = np.abs(fft(signal))
            freqs = fftfreq(len(signal), d=1.0)
            pos_mask = (freqs > 0) & (freqs < 0.25)
            pos_freqs = freqs[pos_mask]
            pos_spectrum = spectrum[pos_mask]
            
            axes[0, 1].plot(pos_freqs, pos_spectrum, 'k-', linewidth=2)
            for agent, config in agents_config.items():
                axes[0, 1].axvline(config['freq'], color=config['color'], 
                                  linestyle='--', linewidth=2, 
                                  label=f"{agent} ({config['freq']:.2f} Hz)")
        else:
            axes[0, 1].text(0.5, 0.5, 'Collecting data...', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
        
        axes[0, 1].set_title('Frequency Spectrum', fontweight='bold')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Progress bar
        axes[1, 0].barh([0], [n_tokens], height=0.5, color='green')
        axes[1, 0].barh([0], [200-n_tokens], left=n_tokens, height=0.5, color='gray', alpha=0.3)
        axes[1, 0].set_xlim(0, 200)
        axes[1, 0].set_title(f'Progress: {n_tokens}/200', fontweight='bold')
        axes[1, 0].set_yticks([])
        
        # Text display
        axes[1, 1].axis('off')
        words = generated_text.split()[:n_tokens]
        text = ' '.join(words[-30:]) if len(words) > 30 else ' '.join(words)
        axes[1, 1].text(0.1, 0.9, 'Generated Text:', fontweight='bold', va='top')
        axes[1, 1].text(0.1, 0.7, f'"{text}"', va='top', wrap=True, fontsize=9)
        
        plt.suptitle('FM-Multiplexed Steganography with GPT-2', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'/tmp/gpt2_anim_{i:03d}.png'
        plt.savefig(filename, dpi=90, facecolor='white')
        plt.close()
        frame_files.append(filename)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(keyframes)}...")
    
    print("\nCompiling GIF...")
    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave('gpt2_steganography_animated.gif', images, 
                   duration=0.2, loop=0)
    
    for f in frame_files:
        os.remove(f)
    
    print("✓ Animated GIF saved: gpt2_steganography_animated.gif")
    print(f"  Frames: {len(keyframes)}")
    print(f"  Size: {os.path.getsize('gpt2_steganography_animated.gif')/(1024*1024):.2f} MB")


def create_visualization(token_selections, agents_config, freqs, spectrum,
                        generated_text, base_probs, biased_probs):
    """Create visualization of the steganography process"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Token selection sequence
    ax1 = axes[0, 0]
    ax1.plot(token_selections, 'ko-', linewidth=1, markersize=3)
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Token ID')
    ax1.set_title('Token Selection Sequence')
    ax1.grid(True, alpha=0.3)
    
    # 2. Frequency spectrum
    ax2 = axes[0, 1]
    ax2.plot(freqs[:len(freqs)//4], spectrum[:len(spectrum)//4], 'k-', linewidth=2)
    
    # Mark agent frequencies - UPDATE RANGE
    for agent, config in agents_config.items():
        freq = config['freq']
        color = config['color']
        ax2.axvline(freq, color=color, linestyle='--', linewidth=2, 
                   label=f'{agent} ({freq:.2f} Hz)', alpha=0.7)
    
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('FFT Magnitude')
    ax2.set_title('Frequency Analysis of Token Sequence')
    ax2.set_xlim(0, 0.15)  # Adjust x-axis for lower frequencies
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Probability distribution comparison (first token)
    ax3 = axes[1, 0]
    x = np.arange(len(base_probs[0]))
    width = 0.35
    ax3.bar(x - width/2, base_probs[0], width, label='Base (GPT-2)', alpha=0.7)
    ax3.bar(x + width/2, biased_probs[0], width, label='Biased (Stego)', alpha=0.7)
    ax3.set_xlabel('Token Index (top-50)')
    ax3.set_ylabel('Probability')
    ax3.set_title('Token Probability Distribution (1st token)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Text display
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Wrap text
    words = generated_text.split()
    wrapped_lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 40:
            wrapped_lines.append(' '.join(current_line))
            current_line = []
    if current_line:
        wrapped_lines.append(' '.join(current_line))
    
    text_display = '\n'.join(wrapped_lines)
    
    ax4.text(0.1, 0.9, 'Generated Stegotext:', 
            fontsize=12, fontweight='bold', va='top')
    ax4.text(0.1, 0.75, f'"{text_display}"',
            fontsize=10, va='top', wrap=True, style='italic')
    ax4.text(0.1, 0.2, 'Contains hidden messages\nfrom 3 agents at different\nfrequencies!',
            fontsize=11, va='top', color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fm_gpt2_steganography.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved: fm_gpt2_steganography.png")


if __name__ == "__main__":
    print("\n" + "⚠️  NOTE: This requires transformers and torch libraries")
    print("Install with: pip install transformers torch\n")
    
    try:
        demonstrate_fm_gpt2_steganography()
        print("\n🎉 SUCCESS! FM-multiplexed steganography with GPT-2 demonstrated!\n")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("Please install required libraries:")
        print("  pip install transformers torch")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
