import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple

@jit
def segment_point_distance(segment_start: jnp.ndarray, 
                          segment_end: jnp.ndarray, 
                          point: jnp.ndarray) -> Tuple[bool, float, jnp.ndarray]:
    """
    Compute shortest distance from a segment to a point.
    
    Args:
        segment_start: Shape (2,) - start point of segment
        segment_end: Shape (2,) - end point of segment  
        point: Shape (2,) - query point
        
    Returns:
        Tuple of:
        - has_orthogonal: bool - True if orthogonal line from point intersects segment
        - distance: float - shortest distance between segment and point
        - normal_vector: Shape (2,) - unit normal vector from segment to point
    """
    # Vector from segment start to end
    segment_vec = segment_end - segment_start
    
    # Vector from segment start to point
    point_vec = point - segment_start
    
    # Length squared of segment (avoid sqrt for efficiency)
    segment_len_sq = jnp.dot(segment_vec, segment_vec)
    
    # Handle degenerate case where segment has zero length
    is_degenerate = segment_len_sq < 1e-12
    
    # Project point onto infinite line containing segment
    # t represents position along segment: 0 = start, 1 = end
    t = jnp.where(is_degenerate, 0.0, jnp.dot(point_vec, segment_vec) / segment_len_sq)
    
    # Check if projection falls within segment bounds [0, 1]
    has_orthogonal = (t >= 0.0) & (t <= 1.0) & (~is_degenerate)
    
    # Clamp t to segment bounds to find closest point on segment
    t_clamped = jnp.clip(t, 0.0, 1.0)
    
    # Find closest point on segment
    closest_point = segment_start + t_clamped * segment_vec
    
    # Vector from closest point to query point
    diff_vec = point - closest_point
    
    # Distance
    distance = jnp.linalg.norm(diff_vec)
    
    # Normal vector (unit vector from segment to point)
    # Handle case where point lies exactly on segment
    normal_vector = jnp.where(distance < 1e-12, 
                             jnp.array([0.0, 0.0]),
                             diff_vec / distance)
    
    return has_orthogonal, distance, normal_vector

@jit 
def batch_segment_point_distance(segments: jnp.ndarray, 
                                points: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute shortest distances from multiple segments to multiple points.
    
    Args:
        segments: Shape (n_segments, 2, 2) - array of segments, each with start and end points
        points: Shape (n_points, 2) - array of query points
        
    Returns:
        Tuple of:
        - has_orthogonal: Shape (n_points, n_segments) - orthogonal intersection flags
        - distances: Shape (n_points, n_segments) - shortest distances  
        - normal_vectors: Shape (n_points, n_segments, 2) - unit normal vectors
    """
    # Use vmap to vectorize over both points and segments
    # First vmap over segments for each point
    segment_vmap = vmap(lambda seg, pt: segment_point_distance(seg[0], seg[1], pt), 
                       in_axes=(0, None), out_axes=0)
    
    # Then vmap over points
    point_segment_vmap = vmap(segment_vmap, in_axes=(None, 0), out_axes=0)
    
    return point_segment_vmap(segments, points)

# Example usage and test functions
def create_test_data():
    """Create sample data for testing."""
    # Define some test segments
    segments = jnp.array([
        [[0.0, 0.0], [1.0, 0.0]],  # horizontal segment
        [[0.0, 0.0], [0.0, 1.0]],  # vertical segment  
        [[0.0, 0.0], [1.0, 1.0]],  # diagonal segment
        [[2.0, 2.0], [3.0, 2.0]],  # another horizontal segment
    ])
    
    # Define some test points
    points = jnp.array([
        [0.5, 1.0],   # above horizontal segment
        [1.0, 0.5],   # right of vertical segment
        [0.0, 0.0],   # on segment start
        [2.5, 3.0],   # above second horizontal segment
        [0.5, 0.5],   # near diagonal segment
    ])
    
    return segments, points

def run_example():
    """Run example computation and display results."""
    segments, points = create_test_data()
    
    print("Segments shape:", segments.shape)
    print("Points shape:", points.shape)
    print()
    
    # Compute distances for all point-segment pairs
    has_ortho, distances, normals = batch_segment_point_distance(segments, points)
    
    print("Results shape - has_orthogonal:", has_ortho.shape)
    print("Results shape - distances:", distances.shape) 
    print("Results shape - normal_vectors:", normals.shape)
    print()
    
    # Display some specific results
    for i in range(len(points)):
        print(f"Point {i}: {points[i]}")
        for j in range(len(segments)):
            seg = segments[j]
            print(f"  Segment {j}: {seg[0]} -> {seg[1]}")
            print(f"    Has orthogonal: {has_ortho[i, j]}")
            print(f"    Distance: {distances[i, j]:.4f}")
            print(f"    Normal: [{normals[i, j, 0]:.4f}, {normals[i, j, 1]:.4f}]")
        print()

if __name__ == "__main__":
    # Run the example
    run_example()
    
    # Test JIT compilation timing
    segments, points = create_test_data()
    
    print("Testing JIT compilation...")
    
    # First call compiles the function
    import time
    start = time.time()
    result1 = batch_segment_point_distance(segments, points)
    compile_time = time.time() - start
    
    # Second call uses compiled version
    start = time.time()
    result2 = batch_segment_point_distance(segments, points)
    run_time = time.time() - start
    
    print(f"First call (with compilation): {compile_time:.4f}s")
    print(f"Second call (compiled): {run_time:.4f}s")
    
    # Verify results are identical
    print("Results identical:", jnp.allclose(result1[1], result2[1]))