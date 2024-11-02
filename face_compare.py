import numpy as np

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

def arcface_similarity(vec1, vec2, margin=0.5):
    cos_theta = cosine_similarity(vec1, vec2)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_with_margin = theta + margin
    return np.cos(theta_with_margin)

def recognize_face(known_face_encodings, face_encoding_to_compare, margin=0.5, threshold=0.5):
    best_match = None
    max_similarity = -1
    
    for person_name, encodings in known_face_encodings.items():
        for known_encoding in encodings:
            similarity = arcface_similarity(known_encoding, face_encoding_to_compare, margin)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = person_name
    
    if max_similarity > threshold:
        return best_match
    return None  
