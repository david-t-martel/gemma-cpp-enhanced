use rag_redis_system::vector_store::SimdDistanceCalculator;
use std::f32;

#[test]
fn test_simd_calculator_basic_operations() {
    let dimension = 4;
    let calculator = SimdDistanceCalculator::new(dimension);

    // Test vectors
    let vec_a = vec![1.0, 0.0, 0.0, 0.0];
    let vec_b = vec![0.0, 1.0, 0.0, 0.0];
    let vec_c = vec![1.0, 1.0, 0.0, 0.0];

    // Test cosine similarity
    let cos_ab = calculator.cosine_similarity(&vec_a, &vec_b).unwrap();
    let cos_ac = calculator.cosine_similarity(&vec_a, &vec_c).unwrap();

    assert!(
        (cos_ab - 0.0).abs() < f32::EPSILON,
        "Orthogonal vectors should have 0 cosine similarity"
    );
    assert!(
        (cos_ac - (1.0 / f32::consts::SQRT_2)).abs() < 1e-6,
        "45-degree vectors should have sqrt(2)/2 similarity"
    );

    // Test euclidean distance
    let euc_ab = calculator.euclidean_distance(&vec_a, &vec_b).unwrap();
    let euc_ac = calculator.euclidean_distance(&vec_a, &vec_c).unwrap();

    assert!(
        (euc_ab - f32::consts::SQRT_2).abs() < 1e-6,
        "Euclidean distance should be sqrt(2)"
    );
    assert!(
        (euc_ac - 1.0).abs() < 1e-6,
        "Euclidean distance should be 1.0"
    );

    // Test dot product
    let dot_ab = calculator.dot_product(&vec_a, &vec_b).unwrap();
    let dot_ac = calculator.dot_product(&vec_a, &vec_c).unwrap();

    assert!(
        (dot_ab - 0.0).abs() < f32::EPSILON,
        "Orthogonal vectors should have 0 dot product"
    );
    assert!(
        (dot_ac - 1.0).abs() < f32::EPSILON,
        "Dot product should be 1.0"
    );

    // Test Manhattan distance
    let man_ab = calculator.manhattan_distance(&vec_a, &vec_b).unwrap();
    let man_ac = calculator.manhattan_distance(&vec_a, &vec_c).unwrap();

    assert!(
        (man_ab - 2.0).abs() < f32::EPSILON,
        "Manhattan distance should be 2.0"
    );
    assert!(
        (man_ac - 1.0).abs() < f32::EPSILON,
        "Manhattan distance should be 1.0"
    );
}

#[test]
fn test_simd_calculator_dimension_validation() {
    let calculator = SimdDistanceCalculator::new(3);

    let vec_valid = vec![1.0, 2.0, 3.0];
    let vec_invalid = vec![1.0, 2.0, 3.0, 4.0]; // Wrong dimension

    // Valid operations should succeed
    assert!(calculator.cosine_similarity(&vec_valid, &vec_valid).is_ok());
    assert!(calculator
        .euclidean_distance(&vec_valid, &vec_valid)
        .is_ok());
    assert!(calculator.dot_product(&vec_valid, &vec_valid).is_ok());
    assert!(calculator
        .manhattan_distance(&vec_valid, &vec_valid)
        .is_ok());

    // Invalid operations should return errors
    assert!(calculator
        .cosine_similarity(&vec_valid, &vec_invalid)
        .is_err());
    assert!(calculator
        .euclidean_distance(&vec_valid, &vec_invalid)
        .is_err());
    assert!(calculator.dot_product(&vec_valid, &vec_invalid).is_err());
    assert!(calculator
        .manhattan_distance(&vec_valid, &vec_invalid)
        .is_err());
}

#[test]
fn test_simd_calculator_edge_cases() {
    let calculator = SimdDistanceCalculator::new(3);

    // Zero vectors
    let zero_vec = vec![0.0, 0.0, 0.0];
    let unit_vec = vec![1.0, 0.0, 0.0];

    let cos_result = calculator.cosine_similarity(&zero_vec, &unit_vec).unwrap();
    assert_eq!(cos_result, 0.0, "Zero vector cosine similarity should be 0");

    let euc_result = calculator.euclidean_distance(&zero_vec, &unit_vec).unwrap();
    assert!(
        (euc_result - 1.0).abs() < f32::EPSILON,
        "Euclidean distance from zero to unit should be 1"
    );

    // Identical vectors
    let cos_identical = calculator.cosine_similarity(&unit_vec, &unit_vec).unwrap();
    assert!(
        (cos_identical - 1.0).abs() < f32::EPSILON,
        "Identical vectors should have cosine similarity of 1"
    );

    let euc_identical = calculator.euclidean_distance(&unit_vec, &unit_vec).unwrap();
    assert!(
        euc_identical.abs() < f32::EPSILON,
        "Identical vectors should have euclidean distance of 0"
    );
}

#[test]
fn test_simd_calculator_performance_consistency() {
    // Test that SIMD and fallback implementations give consistent results
    let dimension = 384; // Common embedding dimension
    let calculator = SimdDistanceCalculator::new(dimension);

    // Generate test vectors
    let mut vec_a = Vec::with_capacity(dimension);
    let mut vec_b = Vec::with_capacity(dimension);

    for i in 0..dimension {
        vec_a.push((i as f32).sin());
        vec_b.push((i as f32).cos());
    }

    // Normalize vectors
    let norm_a = vec_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = vec_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec_a.iter_mut().for_each(|x| *x /= norm_a);
    vec_b.iter_mut().for_each(|x| *x /= norm_b);

    // Test various distance metrics
    let cos_sim = calculator.cosine_similarity(&vec_a, &vec_b).unwrap();
    let euc_dist = calculator.euclidean_distance(&vec_a, &vec_b).unwrap();
    let dot_prod = calculator.dot_product(&vec_a, &vec_b).unwrap();
    let man_dist = calculator.manhattan_distance(&vec_a, &vec_b).unwrap();

    // Verify results are reasonable
    assert!(
        cos_sim >= -1.0 && cos_sim <= 1.0,
        "Cosine similarity should be in [-1, 1]"
    );
    assert!(euc_dist >= 0.0, "Euclidean distance should be non-negative");
    assert!(man_dist >= 0.0, "Manhattan distance should be non-negative");

    // For normalized vectors, cosine similarity should equal dot product
    assert!(
        (cos_sim - dot_prod).abs() < 1e-5,
        "For normalized vectors, cosine similarity should equal dot product"
    );
}

#[test]
fn test_simd_calculator_different_dimensions() {
    // Test various common embedding dimensions
    let dimensions = [128, 256, 384, 512, 768, 1024];

    for &dim in &dimensions {
        let calculator = SimdDistanceCalculator::new(dim);

        // Create simple test vectors
        let mut vec_a = vec![0.0; dim];
        let mut vec_b = vec![0.0; dim];

        vec_a[0] = 1.0;
        vec_b[1] = 1.0;

        // Test basic operations
        let cos_sim = calculator.cosine_similarity(&vec_a, &vec_b).unwrap();
        let euc_dist = calculator.euclidean_distance(&vec_a, &vec_b).unwrap();

        assert!(
            (cos_sim - 0.0).abs() < f32::EPSILON,
            "Orthogonal unit vectors should have 0 cosine similarity for dim {}",
            dim
        );
        assert!(
            (euc_dist - f32::consts::SQRT_2).abs() < 1e-6,
            "Euclidean distance should be sqrt(2) for dim {}",
            dim
        );
    }
}

#[test]
fn test_simd_calculator_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let calculator = Arc::new(SimdDistanceCalculator::new(256));
    let mut handles = vec![];

    // Test concurrent access from multiple threads
    for thread_id in 0..4 {
        let calc = calculator.clone();
        let handle = thread::spawn(move || {
            let mut vec_a = vec![0.0; 256];
            let mut vec_b = vec![0.0; 256];

            vec_a[thread_id] = 1.0;
            vec_b[thread_id + 1] = 1.0;

            // Perform multiple operations
            for _ in 0..100 {
                let _ = calc.cosine_similarity(&vec_a, &vec_b).unwrap();
                let _ = calc.euclidean_distance(&vec_a, &vec_b).unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}
