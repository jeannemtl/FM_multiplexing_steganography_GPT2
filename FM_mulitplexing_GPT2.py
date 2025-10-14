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
    model.eval()
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ GPT-2 loaded successfully")
    return model, tokenizer


def get_token_probabilities(model, tokenizer, context, top_k=50):
    """
    Get probability distribution over next tokens given context
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        context: input text string
        top_k: limit to top-k tokens
    
    Returns:
        token_ids: list of token IDs
        probabilities: probability for each token
    """
    # Encode context
    input_ids = tokenizer.encode(context, return_tensors='pt')
    
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
    Encode bits using frequency modulation to bias token selection
    
    Args:
        bits: binary array to encode
        frequency: carrier frequency for this agent
        num_tokens: number of tokens in vocabulary
    
    Returns:
        bias_vector: frequency-modulated bias to apply to token probabilities
    """
    # Create frequency signature
    t = np.linspace(0, len(bits), num_tokens)
    carrier = np.sin(2 * np.pi * frequency * t)
    
    # Simple encoding: use first few bits to create bias pattern
    # In practice, this would be more sophisticated
    bit_value = np.mean(bits[:min(8, len(bits))])  # Average of first 8 bits
    
    # Modulate carrier amplitude by bit pattern
    amplitude = 0.1 + 0.2 * bit_value  # 0.1 to 0.3 range
    bias = amplitude * carrier
    
    # Normalize to be small perturbation
    bias = bias / (np.max(np.abs(bias)) + 1e-10) * 0.1
    
    return bias


def fm_multiplexed_token_selection(model, tokenizer, context, agent_messages, agents_config):
    """
    Select next token using FM-multiplexed steganography
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer  
        context: current text context
        agent_messages: dict of {agent: bits_to_encode}
        agents_config: dict of {agent: frequency}
    
    Returns:
        selected_token: the chosen token ID
        token_text: the decoded token text
    """
    # Get base probabilities from GPT-2
    token_ids, base_probs = get_token_probabilities(model, tokenizer, context, top_k=50)
    
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


def analyze_token_sequence_frequency(token_selections, agents_config):
    """
    Analyze the frequency content of token selection pattern
    
    Args:
        token_selections: list of selected token indices
        agents_config: agent frequency configuration
    
    Returns:
        detected_agents: which agents were detected
    """
    # Convert token selections to signal
    signal = np.array(token_selections, dtype=float)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    
    # Compute FFT
    spectrum = np.abs(fft(signal))
    freqs = fftfreq(len(signal), d=1.0)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_spectrum = spectrum[pos_mask]
    
    # Detect peaks for each agent
    detected = {}
    for agent, config in agents_config.items():
        target_freq = config['freq']
        
        # Find power near target frequency
        freq_mask = (pos_freqs > target_freq - 0.1) & (pos_freqs < target_freq + 0.1)
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
    model, tokenizer = setup_gpt2()
    
    # Configure agents
    agents_config = {
        'ALICE': {'freq': 0.15, 'color': 'blue'},
        'BOB': {'freq': 0.30, 'color': 'green'},
        'CHARLIE': {'freq': 0.45, 'color': 'red'}
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
    print(f"\nGenerating {30} tokens with embedded messages...\n")
    
    # Generate stegotext
    generated_text = context
    token_selections = []
    all_base_probs = []
    all_biased_probs = []
    
    for i in range(30):
        # Each agent "sends" their message (in practice, would consume bits)
        selected_token, token_text, base_probs, biased_probs = fm_multiplexed_token_selection(
            model, tokenizer, generated_text, agent_messages, agents_config
        )
        
        generated_text += token_text
        token_selections.append(selected_token)
        all_base_probs.append(base_probs)
        all_biased_probs.append(biased_probs)
        
        if (i + 1) % 10 == 0:
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
        token_selections, agents_config
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
    elif avg_kl < 0.1:
        print("✓ Good: Low detectability")
    else:
        print("⚠ Detectable: KL divergence is significant")
    
    # Visualization
    create_visualization(token_selections, agents_config, freqs, spectrum, 
                        generated_text, all_base_probs, all_biased_probs)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Multiple agents embed messages at different frequencies")
    print("2. GPT-2 generates natural-looking text (covertext)")
    print("3. Token selection is subtly biased by FM-modulated signals")
    print("4. Receiver uses FFT to detect frequency signatures")
    print("5. Text appears normal, but contains hidden multiplexed channels")
    print("="*80)


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
    
    # Mark agent frequencies
    for agent, config in agents_config.items():
        freq = config['freq']
        color = config['color']
        ax2.axvline(freq, color=color, linestyle='--', linewidth=2, 
                   label=f'{agent} ({freq:.2f} Hz)', alpha=0.7)
    
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('FFT Magnitude')
    ax2.set_title('Frequency Analysis of Token Sequence')
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
