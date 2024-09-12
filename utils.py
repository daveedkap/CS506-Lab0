import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of the same shape.
    Return the scalar dot product of the two vectors.
    '''
    # Use the numpy dot function to calculate the dot product
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of the same shape.
    Return the cosine similarity between the two vectors.
    '''
    # Use dot_product function to get the dot product of v1 and v2
    dot_prod = dot_product(v1, v2)
    
    # Calculate the norms (Euclidean length) of v1 and v2 using np.linalg.norm
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Return the cosine similarity: (v1 dot v2) / (||v1|| * ||v2||)
    return dot_prod / (norm_v1 * norm_v2)

def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    Return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    '''
    # Initialize variables to store the highest cosine similarity and the index
    max_similarity = -1  # Cosine similarity ranges from -1 to 1
    best_index = -1

    # Loop through each vector in the matrix
    for i in range(len(vectors)):
        # Compute cosine similarity between target_vector and current vector
        similarity = cosine_similarity(target_vector, vectors[i])

        # Update the maximum similarity and index if the current similarity is higher
        if similarity > max_similarity:
            max_similarity = similarity
            best_index = i

    return best_index