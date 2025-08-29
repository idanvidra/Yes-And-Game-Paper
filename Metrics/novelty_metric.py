"""
Novelty calculation functions using embedding-based approaches.
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from data_utils import (
    load_stories_from_json, prepare_hh_data, prepare_aih_data,
    extract_sentences_from_story, get_speaker_labels_for_story
)

def compute_consecutive_distances(sentences, model):
    """
    Compute consecutive sentence distances for a list of sentences.
    Higher distances indicate greater semantic novelty between successive sentences.
    
    Args:
        sentences: List of sentences
        model: SentenceTransformer model
        
    Returns:
        numpy array of distances between consecutive sentences
    """
    if len(sentences) < 2:
        return np.array([])

    embeddings = model.encode(sentences)
    distances = cosine_distances(embeddings[:-1], embeddings[1:]).diagonal()
    return distances

def analyze_consecutive_novelty(hh_stories, aih_stories, model_name='all-MiniLM-L6-v2', sentence_filter=2):
    """
    Analyze consecutive sentence novelty within each story.
    For each story, compute distances between consecutive sentences and label by speaker.
    Distances are attributed to the RESPONDING speaker (the second in each pair).
    
    Args:
        hh_stories: List of HH story dictionaries
        aih_stories: List of AIH story dictionaries
        model_name: SentenceTransformer model name
        sentence_filter: Minimum number of sentences required
        
    Returns:
        dict: Results for H_HH, H_AIH, and AI_AIH groups
    """
    model = SentenceTransformer(model_name)
    
    # Initialize results
    h_hh_distances = []
    h_aih_distances = []
    ai_aih_distances = []
    
    # Process HH stories (all sentences are human)
    for story in hh_stories:
        sentences = extract_sentences_from_story(story)
        if len(sentences) > sentence_filter:
            # All sentences in HH are human
            story_distances = compute_consecutive_distances(sentences, model)
            h_hh_distances.extend(story_distances)
    
    # Process AIH stories (alternating human/AI)
    for story in aih_stories:
        sentences = extract_sentences_from_story(story)
        if len(sentences) > sentence_filter:
            # Get speaker labels for this story
            speaker_labels = get_speaker_labels_for_story(story)
            
            # Compute distances between consecutive sentences
            story_distances = compute_consecutive_distances(sentences, model)
            
            # Assign distances to the RESPONDING speaker (second in the pair)
            for i, distance in enumerate(story_distances):
                responding_speaker = speaker_labels[i + 1]
                if responding_speaker == 'H_AIH':
                    h_aih_distances.append(distance)
                elif responding_speaker == 'AI_AIH':
                    ai_aih_distances.append(distance)
    
    results = {
        'H_HH': np.array(h_hh_distances),
        'H_AIH': np.array(h_aih_distances),
        'AI_AIH': np.array(ai_aih_distances)
    }
    
    return results, model_name

def compute_story_novelty_scores(stories, model):
    """
    Compute novelty scores for entire stories.
    Novelty = 1 - average similarity to all other stories.
    
    Args:
        stories: List of story sentence lists
        model: SentenceTransformer model
        
    Returns:
        List of novelty scores
    """
    if len(stories) < 2:
        return []
    
    # Compute story embeddings (average of sentence embeddings)
    story_embeddings = []
    for story in stories:
        if len(story) == 0:
            story_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
        else:
            sentence_embeddings = model.encode(story)
            story_embedding = np.mean(sentence_embeddings, axis=0)
            story_embeddings.append(story_embedding)
    
    story_embeddings = np.array(story_embeddings)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(story_embeddings)
    
    # Calculate novelty scores
    novelty_scores = []
    for i in range(len(stories)):
        # Exclude similarity to self
        others_sim = np.delete(sim_matrix[i], i)
        avg_similarity = np.mean(others_sim) if len(others_sim) > 0 else 0
        novelty = 1 - avg_similarity
        novelty_scores.append(novelty)
    
    return novelty_scores

def analyze_story_novelty(hh_stories, aih_stories, model_name='all-MiniLM-L6-v2', sentence_filter=2):
    """
    Analyze story-level novelty between HH and AIH conditions.
    
    Args:
        hh_stories: List of HH story dictionaries
        aih_stories: List of AIH story dictionaries
        model_name: SentenceTransformer model name
        sentence_filter: Minimum number of sentences required
        
    Returns:
        dict: Novelty scores for HH and AIH groups
    """
    model = SentenceTransformer(model_name)
    
    # Prepare data
    hh_filtered_stories, _ = prepare_hh_data(hh_stories, sentence_filter)
    aih_filtered_stories, _, _ = prepare_aih_data(aih_stories, sentence_filter)
    
    hh_novelty = compute_story_novelty_scores(hh_filtered_stories, model)
    aih_novelty = compute_story_novelty_scores(aih_filtered_stories, model)
    
    return {
        'HH': np.array(hh_novelty),
        'AIH': np.array(aih_novelty)
    }, model_name

def analyze_novelty_for_multiple_models(models, hh_stories, aih_stories, sentence_filter=2):
    """
    Analyze novelty across multiple embedding models.
    
    Args:
        models: List of model names to test
        hh_stories: List of HH story dictionaries
        aih_stories: List of AIH story dictionaries
        sentence_filter: Minimum number of sentences required
        
    Returns:
        dict: Results for each model
    """
    all_results = {}
    
    for model_name in models:
        print(f"Analyzing model: {model_name}")
        
        # Consecutive novelty
        consecutive_results, _ = analyze_consecutive_novelty(
            hh_stories, aih_stories, model_name, sentence_filter
        )
        
        # Story-level novelty
        story_results, _ = analyze_story_novelty(
            hh_stories, aih_stories, model_name, sentence_filter
        )
        
        all_results[model_name] = {
            'consecutive': consecutive_results,
            'story_level': story_results
        }
    
    return all_results

def get_novelty_statistics(results):
    """
    Compute statistics for novelty results.
    
    Args:
        results: Results from analyze_consecutive_novelty or analyze_story_novelty
        
    Returns:
        dict: Statistics for each group
    """
    stats = {}
    
    for group, values in results.items():
        if len(values) > 0:
            stats[group] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        else:
            stats[group] = {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan
            }
    
    return stats

def analyze_speaker_transition_novelty(aih_stories, model_name='all-MiniLM-L6-v2', sentence_filter=2):
    """
    Analyze novelty at speaker transitions in AIH stories.
    This measures how much each speaker varies from the previous speaker's contribution.
    
    Args:
        aih_stories: List of AIH story dictionaries
        model_name: SentenceTransformer model name
        sentence_filter: Minimum number of sentences required
        
    Returns:
        dict: Results for human-to-AI and AI-to-human transitions
    """
    model = SentenceTransformer(model_name)
    
    human_to_ai_distances = []
    ai_to_human_distances = []
    
    for story in aih_stories:
        sentences = extract_sentences_from_story(story)
        if len(sentences) > sentence_filter:
            speaker_labels = get_speaker_labels_for_story(story)
            
            # Compute distances between consecutive sentences
            story_distances = compute_consecutive_distances(sentences, model)
            
            # Analyze transitions
            for i, distance in enumerate(story_distances):
                current_speaker = speaker_labels[i]
                next_speaker = speaker_labels[i + 1]
                
                if current_speaker == 'H_AIH' and next_speaker == 'AI_AIH':
                    # Human followed by AI
                    human_to_ai_distances.append(distance)
                elif current_speaker == 'AI_AIH' and next_speaker == 'H_AIH':
                    # AI followed by Human
                    ai_to_human_distances.append(distance)
    
    results = {
        'Human_to_AI': np.array(human_to_ai_distances),
        'AI_to_Human': np.array(ai_to_human_distances)
    }
    
    return results, model_name
    

if __name__ == "__main__":
    # Use smaller model for local testing
    SMALL_MODEL = 'all-MiniLM-L6-v2'  # 80MB vs 420MB, much faster for local use
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Construct absolute paths to the JSON files
    hh_path = os.path.join(project_root, "Story Datasets", "hh_stories.json")
    aih_path = os.path.join(project_root, "Story Datasets", "aih_stories.json")
    
    # Load all stories
    hh_stories = load_stories_from_json(hh_path)
    aih_stories = load_stories_from_json(aih_path)
    
    print(f"Loaded {len(hh_stories)} HH stories and {len(aih_stories)} AIH stories")
    
    # Test on single examples from each type
    print("\n" + "="*50)
    print("TESTING ON SINGLE EXAMPLES")
    print("="*50)
    
    # Get first story from each type and extract sentences
    hh_example = extract_sentences_from_story(hh_stories[0]) if hh_stories else []
    aih_example = extract_sentences_from_story(aih_stories[0]) if aih_stories else []
    
    print(f"\nHH Example Story (first {min(5, len(hh_example))} sentences):")
    for i, sentence in enumerate(hh_example[:5]):
        print(f"  {i+1}. {sentence}")
    
    print(f"\nAIH Example Story (first {min(5, len(aih_example))} sentences):")
    for i, sentence in enumerate(aih_example[:5]):
        print(f"  {i+1}. {sentence}")
    
    # Test consecutive novelty on single examples
    print("\n" + "="*50)
    print("CONSECUTIVE NOVELTY ANALYSIS")
    print("="*50)
    
    model = SentenceTransformer(SMALL_MODEL)
    print(f"Using model: {SMALL_MODEL}")
    
    # Test consecutive distances for single stories
    if len(hh_example) >= 2:
        hh_distances = compute_consecutive_distances(hh_example, model)
        print(f"\nHH Example consecutive distances: {hh_distances[:5]}")  # Show first 5
        print(f"HH Example mean distance: {np.mean(hh_distances):.4f}")
    
    if len(aih_example) >= 2:
        aih_distances = compute_consecutive_distances(aih_example, model)
        print(f"AIH Example consecutive distances: {aih_distances[:5]}")  # Show first 5
        print(f"AIH Example mean distance: {np.mean(aih_distances):.4f}")
    
    # Test full consecutive novelty analysis
    print("\n" + "="*50)
    print("FULL CONSECUTIVE NOVELTY ANALYSIS")
    print("="*50)
    
    consecutive_results, _ = analyze_consecutive_novelty(hh_stories, aih_stories, SMALL_MODEL, sentence_filter=2)
    consecutive_stats = get_novelty_statistics(consecutive_results)
    
    for group, stats in consecutive_stats.items():
        print(f"\n{group}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
    
    # Test speaker transition novelty (more meaningful for AIH)
    print("\n" + "="*50)
    print("SPEAKER TRANSITION NOVELTY ANALYSIS")
    print("="*50)
    
    transition_results, _ = analyze_speaker_transition_novelty(aih_stories, SMALL_MODEL, sentence_filter=2)
    transition_stats = get_novelty_statistics(transition_results)
    
    for group, stats in transition_stats.items():
        print(f"\n{group}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
    
    # Test story-level novelty
    print("\n" + "="*50)
    print("STORY-LEVEL NOVELTY ANALYSIS")
    print("="*50)
    
    story_results, _ = analyze_story_novelty(hh_stories, aih_stories, SMALL_MODEL, sentence_filter=2)
    story_stats = get_novelty_statistics(story_results)
    
    for group, stats in story_stats.items():
        print(f"\n{group}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
    
    # Test multiple models (just the small one for now)
    print("\n" + "="*50)
    print("MULTIPLE MODELS ANALYSIS")
    print("="*50)
    
    models_to_test = [SMALL_MODEL]  # Add more models here if needed
    multi_results = analyze_novelty_for_multiple_models(models_to_test, hh_stories, aih_stories, sentence_filter=2)
    
    print(f"\nCompleted analysis for {len(models_to_test)} model(s)")
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*50)