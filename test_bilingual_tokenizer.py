"""
Example script demonstrating the improved BilingualTokenizer for English and Spanish.
This shows both training and inference workflows.
"""

import torch
from simpletokenizer import BilingualTokenizer

def demonstrate_basic_usage():
    """Demonstrate basic tokenizer usage."""
    print("=" * 70)
    print("BASIC USAGE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize tokenizer
    tokenizer = BilingualTokenizer(
        max_vocab_size=65536,
        embedding_dim=300,
        use_language_tokens=True,
        multilingual_vocab=True
    )
    
    print("\n1. Building vocabulary from bilingual texts...")
    
    # Training data
    texts = [
        "Hello, how are you today?",
        "The weather is beautiful this morning.",
        "I love learning new languages.",
        "Hola, ¿cómo estás hoy?",
        "El clima es hermoso esta mañana.",
        "Me encanta aprender nuevos idiomas.",
        "Good morning, nice to meet you!",
        "Buenos días, mucho gusto conocerte!",
    ]
    
    languages = ['en', 'en', 'en', 'es', 'es', 'es', 'en', 'es']
    
    # Build vocabulary
    tokenizer.fit(texts, languages, remove_accents_flag=False)
    
    print(f"✓ Vocabulary size: {tokenizer.get_vocabulary_size()}")
    print(f"✓ Language stats: {tokenizer.get_language_stats()}")
    
    print("\n2. Encoding and decoding single text...")
    
    # English example
    english_text = "How are you doing today?"
    english_tokens = tokenizer.encode(english_text)
    decoded_en = tokenizer.decode(english_tokens, skip_special_tokens=True)
    
    print(f"Original EN:  '{english_text}'")
    print(f"Tokens:       {english_tokens}")
    print(f"Decoded EN:   '{decoded_en}'")
    
    # Spanish example
    spanish_text = "¿Cómo estás hoy?"
    spanish_tokens = tokenizer.encode(spanish_text)
    decoded_es = tokenizer.decode(spanish_tokens, skip_special_tokens=True)
    
    print(f"\nOriginal ES:  '{spanish_text}'")
    print(f"Tokens:       {spanish_tokens}")
    print(f"Decoded ES:   '{decoded_es}'")
    
    print("\n3. Language detection...")
    
    test_texts = [
        "The quick brown fox",
        "El rápido zorro marrón",
        "I am learning Spanish",
        "Estoy aprendiendo inglés"
    ]
    
    for text in test_texts:
        detected_lang = tokenizer.detect_language(text)
        print(f"'{text}' → Detected: {detected_lang}")


def demonstrate_batch_processing():
    """Demonstrate batch encoding and decoding."""
    print("\n" + "=" * 70)
    print("BATCH PROCESSING DEMONSTRATION")
    print("=" * 70)
    
    tokenizer = BilingualTokenizer(max_vocab_size=65536, embedding_dim=300)
    
    # Build vocabulary
    training_texts = [
        "Hello world", "Hola mundo",
        "Good morning", "Buenos días",
        "Thank you", "Gracias"
    ]
    training_langs = ['en', 'es', 'en', 'es', 'en', 'es']
    tokenizer.fit(training_texts, training_langs)
    
    print("\n1. Batch encoding with padding...")
    
    batch_texts = [
        "Hello, how are you?",
        "¿Cómo estás tú hoy?",
        "Good morning everyone!",
        "¡Buenos días a todos!"
    ]
    
    # Batch encode
    batch_tensor = tokenizer.batch_encode(
        texts=batch_texts,
        add_language_token=True,
        max_length=20,
        pad=True
    )
    
    print(f"Batch shape: {batch_tensor.shape}")
    print(f"Batch tensor:\n{batch_tensor}")
    
    print("\n2. Batch decoding...")
    
    # Batch decode
    decoded_batch = tokenizer.batch_decode(batch_tensor, skip_special_tokens=True)
    
    for i, (original, decoded) in enumerate(zip(batch_texts, decoded_batch)):
        print(f"{i+1}. Original: {original}")
        print(f"   Decoded:  {decoded}")


def demonstrate_preprocessing():
    """Demonstrate text preprocessing capabilities."""
    print("\n" + "=" * 70)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 70)
    
    tokenizer = BilingualTokenizer()
    
    print("\n1. English contractions and punctuation...")
    
    english_examples = [
        "I'm learning Spanish and it's great!",
        "They're going to visit us next week.",
        "Don't worry, won't take too long.",
    ]
    
    for text in english_examples:
        cleaned = tokenizer.preprocess_text(text, language='en')
        print(f"Input:   {text}")
        print(f"Output:  {cleaned}\n")
    
    print("2. Spanish accent handling...")
    
    spanish_examples = [
        "¿Cómo estás? ¡Muy bien, gracias!",
        "El niño comió manzanas verdes.",
        "Usaré la llave para abrir la puerta.",
    ]
    
    for text in spanish_examples:
        # With accents
        cleaned_with = tokenizer.preprocess_text(text, language='es', remove_accents_flag=False)
        # Without accents
        cleaned_without = tokenizer.preprocess_text(text, language='es', remove_accents_flag=True)
        
        print(f"Input:           {text}")
        print(f"With accents:    {cleaned_with}")
        print(f"Without accents: {cleaned_without}\n")


def demonstrate_vocabulary_management():
    """Demonstrate vocabulary management features."""
    print("\n" + "=" * 70)
    print("VOCABULARY MANAGEMENT DEMONSTRATION")
    print("=" * 70)
    
    tokenizer = BilingualTokenizer(max_vocab_size=1000, embedding_dim=300)
    
    # Build initial vocabulary
    texts = ["Hello world", "Hola mundo", "Good morning", "Buenos días"]
    langs = ['en', 'es', 'en', 'es']
    tokenizer.fit(texts, langs)
    
    print(f"\n1. Initial vocabulary size: {tokenizer.get_vocabulary_size()}")
    
    print("\n2. Getting vocabulary dictionary...")
    vocab = tokenizer.get_vocab_dict()
    print(f"First 15 vocab items:")
    for i, (word, idx) in enumerate(list(vocab.items())[:15]):
        print(f"  {idx:3d} → {word}")
    
    print("\n3. Adding new words...")
    new_words = ["chatbot", "intelligence", "conversation"]
    for word in new_words:
        idx = tokenizer.add_word(word, language='en')
        print(f"Added '{word}' with index {idx}")
    
    print(f"\nVocabulary size after additions: {tokenizer.get_vocabulary_size()}")
    
    print("\n4. Getting word embeddings...")
    embedding = tokenizer.get_word_embedding("hello")
    print(f"Word 'hello' embedding shape: {embedding.shape}")
    print(f"First 10 dimensions: {embedding[:10]}")


def demonstrate_special_tokens():
    """Demonstrate special tokens usage."""
    print("\n" + "=" * 70)
    print("SPECIAL TOKENS DEMONSTRATION")
    print("=" * 70)
    
    tokenizer = BilingualTokenizer(use_language_tokens=True)
    texts = ["Hello", "Hola", "Good", "Bien"]
    langs = ['en', 'es', 'en', 'es']
    tokenizer.fit(texts, langs)
    
    print("\n1. Special token indices...")
    print(f"<PAD>    = {tokenizer.get_pad_index()}")
    print(f"<UNK>    = {tokenizer.get_unknown_index()}")
    print(f"<START>  = {tokenizer.trie.get_index('<START>')}")
    print(f"<END>    = {tokenizer.trie.get_index('<END>')}")
    print(f"<EN>     = {tokenizer.get_language_token_index('en')}")
    print(f"<ES>     = {tokenizer.get_language_token_index('es')}")
    
    print("\n2. Encoding with special tokens...")
    text = "Hello world"
    
    # With language token
    with_lang = tokenizer.encode(text, add_language_token=True)
    without_lang = tokenizer.encode(text, add_language_token=False)
    
    print(f"With language token:    {with_lang}")
    print(f"Without language token: {without_lang}")
    
    print("\n3. Decoding with/without special tokens...")
    decoded_all = tokenizer.decode(with_lang, skip_special_tokens=False)
    decoded_filtered = tokenizer.decode(with_lang, skip_special_tokens=True)
    
    print(f"All tokens:    '{decoded_all}'")
    print(f"Filtered:      '{decoded_filtered}'")


def demonstrate_training_workflow():
    """Demonstrate a typical training workflow."""
    print("\n" + "=" * 70)
    print("TRAINING WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. Initialize tokenizer...")
    tokenizer = BilingualTokenizer(
        max_vocab_size=65536,
        embedding_dim=300,
        use_language_tokens=True
    )
    
    print("✓ Tokenizer initialized")
    
    print("\n2. Prepare and load training data...")
    # Simulated training data
    train_data = [
        ("Hello, I'm excited to chat", 'en'),
        ("Good morning, how can I help you?", 'en'),
        ("What's your name?", 'en'),
        ("Hola, estoy emocionado de conversar", 'es'),
        ("Buenos días, ¿cómo puedo ayudarte?", 'es'),
        ("¿Cuál es tu nombre?", 'es'),
    ]
    
    train_texts = [t[0] for t in train_data]
    train_langs = [t[1] for t in train_data]
    
    print(f"✓ Loaded {len(train_texts)} training samples")
    print(f"  - {len([l for l in train_langs if l == 'en'])} English samples")
    print(f"  - {len([l for l in train_langs if l == 'es'])} Spanish samples")
    
    print("\n3. Build vocabulary...")
    tokenizer.fit(train_texts, train_langs, remove_accents_flag=False)
    print(f"✓ Vocabulary built with {tokenizer.get_vocabulary_size()} tokens")
    
    print("\n4. Prepare batch for training...")
    batch_data = [
        "How are you today?",
        "¿Cómo estás hoy?"
    ]
    
    batch_tensor = tokenizer.batch_encode(
        texts=batch_data,
        max_length=50,
        pad=True
    )
    
    print(f"✓ Batch tensor shape: {batch_tensor.shape}")
    
    print("\n5. Get embeddings for model...")
    embeddings = tokenizer.get_embeddings()(batch_tensor)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    print(f"  (batch_size={batch_tensor.shape[0]}, seq_length={batch_tensor.shape[1]}, embedding_dim={embeddings.shape[2]})")


def demonstrate_inference_workflow():
    """Demonstrate a typical inference workflow."""
    print("\n" + "=" * 70)
    print("INFERENCE WORKFLOW DEMONSTRATION")
    print("=" * 70)
    
    # Assume model is already trained
    print("\n1. Load pre-trained tokenizer...")
    tokenizer = BilingualTokenizer(
        max_vocab_size=65536,
        embedding_dim=300,
        use_language_tokens=True
    )
    
    # In real scenario, would load: tokenizer.load_vocabulary('vocab.json')
    texts = ["Hello", "Hola", "How are you", "¿Cómo estás?"]
    langs = ['en', 'es', 'en', 'es']
    tokenizer.fit(texts, langs)
    
    print("✓ Tokenizer loaded")
    
    print("\n2. Get user input...")
    user_inputs = [
        "Hello, how are you doing?",
        "¿Hola, cómo estás tú?",
    ]
    
    for user_input in user_inputs:
        print(f"\nUser: {user_input}")
        
        print("\n3. Preprocess and tokenize...")
        detected_lang = tokenizer.detect_language(user_input)
        tokens = tokenizer.encode(user_input, add_language_token=True)
        print(f"  Detected language: {detected_lang}")
        print(f"  Tokenized: {tokens}")
        
        print("\n4. Get embeddings for model...")
        token_tensor = torch.tensor([tokens], dtype=torch.long)
        embeddings = tokenizer.get_embeddings()(token_tensor)
        print(f"  Embeddings shape: {embeddings.shape}")
        
        print("\n5. Model generates response (simulated)...")
        # Simulated model output
        response_tokens = [2, 4, 11, 13, 15, 3]  # Simulated output tokens
        
        print("\n6. Decode response...")
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        print(f"Bot: {response}")


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  BILINGUAL TOKENIZER (English & Spanish) - EXAMPLES".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Run all demonstrations
    demonstrate_basic_usage()
    demonstrate_batch_processing()
    demonstrate_preprocessing()
    demonstrate_vocabulary_management()
    demonstrate_special_tokens()
    demonstrate_training_workflow()
    demonstrate_inference_workflow()
    
    print("\n" + "█" * 70)
    print("█" + "  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!".center(68) + "█")
    print("█" * 70 + "\n")
