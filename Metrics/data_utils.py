"""
Data loading and preprocessing utilities for Yes-And-Game dataset.
"""
import json
import numpy as np

def load_stories_from_json(file_path):
    """
    Load stories from JSON file.
    
    Args:
        file_path: Path to JSON file containing stories
        
    Returns:
        stories: List of story dictionaries with 'id', 'messages', etc.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_sentences_from_story(story):
    """
    Extract sentences from a story dictionary.
    
    Args:
        story: Dictionary with 'messages' key containing list of message dicts
        
    Returns:
        sentences: List of sentence strings
    """
    sentences = []
    for message in story['messages']:
        content = message['content'].strip()
        if content and content != "________________":  # Skip empty or separator messages
            sentences.append(content)
    return sentences

def extract_sentences_by_role(story):
    """
    Extract sentences grouped by role from a story.
    
    Args:
        story: Dictionary with 'messages' key containing list of message dicts
        
    Returns:
        human_sentences: List of human sentences
        ai_sentences: List of AI sentences (for AIH stories)
    """
    human_sentences = []
    ai_sentences = []
    
    for message in story['messages']:
        content = message['content'].strip()
        if content and content != "________________":
            if message['role'] in ['human', 'human1', 'human2']:
                human_sentences.append(content)
            elif message['role'] == 'ai':
                ai_sentences.append(content)
    
    return human_sentences, ai_sentences

def prepare_hh_data(hh_stories, sentence_filter=None):
    """
    Prepare HH data for analysis.
    All sentences are labeled as H_HH.
    
    Args:
        hh_stories: List of HH story dictionaries
        sentence_filter: Minimum number of sentences required (optional)
        
    Returns:
        filtered_stories: List of story sentence lists
        all_sentences: List of all sentences from all stories
    """
    filtered_stories = []
    all_sentences = []
    
    for story in hh_stories:
        sentences = extract_sentences_from_story(story)
        if sentence_filter is None or len(sentences) > sentence_filter:
            filtered_stories.append(sentences)
            all_sentences.extend(sentences)
    
    return filtered_stories, all_sentences

def prepare_aih_data(aih_stories, sentence_filter=None):
    """
    Prepare AIH data for analysis.
    Separates human and AI sentences.
    
    Args:
        aih_stories: List of AIH story dictionaries
        sentence_filter: Minimum number of sentences required (optional)
        
    Returns:
        filtered_stories: List of story sentence lists
        human_sentences: List of all human sentences
        ai_sentences: List of all AI sentences
    """
    filtered_stories = []
    human_sentences = []
    ai_sentences = []
    
    for story in aih_stories:
        sentences = extract_sentences_from_story(story)
        if sentence_filter is None or len(sentences) > sentence_filter:
            filtered_stories.append(sentences)
            
            # Separate by role
            story_human, story_ai = extract_sentences_by_role(story)
            human_sentences.extend(story_human)
            ai_sentences.extend(story_ai)
    
    return filtered_stories, human_sentences, ai_sentences

def get_speaker_labels(story_type, story_length):
    """
    Generate speaker labels for a story.
    
    Args:
        story_type: 'hh' or 'aih'
        story_length: Number of sentences in the story
    
    Returns:
        List of labels: ['H_HH', 'H_HH', ...] or ['H_AIH', 'AI_AIH', 'H_AIH', ...]
    """
    if story_type == 'hh':
        return ['H_HH'] * story_length
    elif story_type == 'aih':
        return ['H_AIH' if i % 2 == 0 else 'AI_AIH' for i in range(story_length)]
    else:
        raise ValueError("story_type must be 'hh' or 'aih'")

def get_speaker_labels_for_story(story):
    """
    Generate speaker labels for a specific story based on its messages.
    
    Args:
        story: Story dictionary with 'messages' key
        
    Returns:
        List of speaker labels for each sentence
    """
    labels = []
    for message in story['messages']:
        content = message['content'].strip()
        if content and content != "________________":
            if message['role'] in ['human', 'human1', 'human2']:
                labels.append('H_AIH' if message['role'] == 'human' else 'H_HH')
            elif message['role'] == 'ai':
                labels.append('AI_AIH')
    return labels

def get_consecutive_sentence_pairs(stories):
    """
    Get consecutive sentence pairs for novelty analysis.
    
    Args:
        stories: List of story dictionaries
        
    Returns:
        pairs: List of (sentence1, sentence2) tuples
        labels: List of corresponding speaker labels
    """
    pairs = []
    labels = []
    
    for story in stories:
        sentences = extract_sentences_from_story(story)
        story_labels = get_speaker_labels_for_story(story)
        
        for i in range(len(sentences) - 1):
            pairs.append((sentences[i], sentences[i + 1]))
            labels.append(story_labels[i])  # Label of the first sentence in pair
    
    return pairs, labels

def get_story_statistics(stories):
    """
    Get basic statistics about stories.
    
    Args:
        stories: List of story dictionaries
        
    Returns:
        dict: Statistics including counts, lengths, etc.
    """
    total_stories = len(stories)
    total_sentences = 0
    story_lengths = []
    
    for story in stories:
        sentences = extract_sentences_from_story(story)
        total_sentences += len(sentences)
        story_lengths.append(len(sentences))
    
    return {
        'total_stories': total_stories,
        'total_sentences': total_sentences,
        'avg_story_length': np.mean(story_lengths) if story_lengths else 0,
        'min_story_length': min(story_lengths) if story_lengths else 0,
        'max_story_length': max(story_lengths) if story_lengths else 0,
        'story_lengths': story_lengths
    }
    
if __name__ == "__main__":
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    
    # Construct absolute paths to the JSON files
    hh_path = os.path.join(project_root, "Story Datasets", "hh_stories.json")
    aih_path = os.path.join(project_root, "Story Datasets", "aih_stories.json")
    
    # Load raw stories
    hh_stories = load_stories_from_json(hh_path)
    aih_stories = load_stories_from_json(aih_path)
    
    print(f"Loaded {len(hh_stories)} HH stories and {len(aih_stories)} AIH stories")
    
    # Get statistics
    hh_stats = get_story_statistics(hh_stories)
    aih_stats = get_story_statistics(aih_stories)
    
    print(f"\nHH Statistics:")
    print(f"  Total stories: {hh_stats['total_stories']}")
    print(f"  Total sentences: {hh_stats['total_sentences']}")
    print(f"  Average story length: {hh_stats['avg_story_length']:.1f} sentences")
    
    print(f"\nAIH Statistics:")
    print(f"  Total stories: {aih_stats['total_stories']}")
    print(f"  Total sentences: {aih_stats['total_sentences']}")
    print(f"  Average story length: {aih_stats['avg_story_length']:.1f} sentences")
    
    # Test data preparation
    hh_filtered_stories, hh_all_sentences = prepare_hh_data(hh_stories, sentence_filter=2)
    aih_filtered_stories, aih_human_sentences, aih_ai_sentences = prepare_aih_data(aih_stories, sentence_filter=2)
    
    print(f"\nAfter filtering (min 3 sentences):")
    print(f"  HH: {len(hh_filtered_stories)} stories, {len(hh_all_sentences)} sentences")
    print(f"  AIH: {len(aih_filtered_stories)} stories, {len(aih_human_sentences)} human sentences, {len(aih_ai_sentences)} AI sentences")
    
    # Show example
    if hh_filtered_stories:
        print(f"\nExample HH story (first 3 sentences):")
        example = hh_filtered_stories[0][:3]
        for i, sentence in enumerate(example):
            print(f"  {i+1}. {sentence}")
    
    if aih_filtered_stories:
        print(f"\nExample AIH story (first 3 sentences):")
        example = aih_filtered_stories[0][:3]
        for i, sentence in enumerate(example):
            print(f"  {i+1}. {sentence}")