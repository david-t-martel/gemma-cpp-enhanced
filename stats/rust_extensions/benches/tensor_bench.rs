use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gemma_extensions::tensor_ops::{simd_dot_product, simd_vector_add, TensorOperations};

fn simd_dot_product_benchmark(c: &mut Criterion) {
    let sizes = vec![64, 256, 1024, 4096, 16384];

    let mut group = c.benchmark_group("simd_dot_product");

    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.1).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(format!("size_{}", size), &size, |bench, _| {
            bench.iter(|| simd_dot_product(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn vector_operations_benchmark(c: &mut Criterion) {
    let size = 4096;
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32 * 0.1).collect();

    let mut group = c.benchmark_group("vector_operations");
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("simd_vector_add", |bench| {
        bench.iter(|| simd_vector_add(black_box(&a), black_box(&b)))
    });

    // Scalar baseline
    group.bench_function("scalar_vector_add", |bench| {
        bench.iter(|| {
            let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
            black_box(result)
        })
    });

    group.finish();
}

fn matrix_multiplication_benchmark(c: &mut Criterion) {
    let tensor_ops = TensorOperations::new(None);
    let sizes = vec![(64, 64, 64), (128, 128, 128), (256, 256, 256)];

    let mut group = c.benchmark_group("matrix_multiplication");

    for (m, k, n) in sizes {
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01).collect();

        group.throughput(Throughput::Elements((m * n) as u64));
        group.bench_with_input(format!("{}x{}x{}", m, k, n), &(m, k, n), |bench, _| {
            bench.iter(|| tensor_ops.matmul(black_box(a.clone()), black_box(b.clone()), m, k, n))
        });
    }

    group.finish();
}

fn activation_functions_benchmark(c: &mut Criterion) {
    let tensor_ops = TensorOperations::new(None);
    let size = 4096;
    let input: Vec<f32> = (0..size)
        .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
        .collect();

    let mut group = c.benchmark_group("activation_functions");
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("gelu", |bench| {
        bench.iter(|| tensor_ops.gelu(black_box(input.clone())))
    });

    group.bench_function("silu", |bench| {
        bench.iter(|| tensor_ops.silu(black_box(input.clone())))
    });

    group.bench_function("softmax", |bench| {
        bench.iter(|| tensor_ops.softmax(black_box(input.clone())))
    });

    group.finish();
}

fn attention_benchmark(c: &mut Criterion) {
    let tensor_ops = TensorOperations::new(None);
    let seq_len = 128;
    let hidden_dim = 512;

    let query: Vec<f32> = (0..seq_len * hidden_dim).map(|i| i as f32 * 0.01).collect();
    let key: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i + 1) as f32 * 0.01)
        .collect();
    let value: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i + 2) as f32 * 0.01)
        .collect();

    let mut group = c.benchmark_group("attention");
    group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));

    group.bench_function("scaled_dot_product_attention", |bench| {
        bench.iter(|| {
            tensor_ops.scaled_dot_product_attention(
                black_box(query.clone()),
                black_box(key.clone()),
                black_box(value.clone()),
                seq_len,
                hidden_dim,
                None,
            )
        })
    });

    group.finish();
}

fn layer_norm_benchmark(c: &mut Criterion) {
    let tensor_ops = TensorOperations::new(None);
    let size = 4096;

    let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
    let gamma: Vec<f32> = vec![1.0; size];
    let beta: Vec<f32> = vec![0.0; size];

    let mut group = c.benchmark_group("layer_norm");
    group.throughput(Throughput::Elements(size as u64));

    group.bench_function("layer_norm", |bench| {
        bench.iter(|| {
            tensor_ops.layer_norm(
                black_box(input.clone()),
                black_box(gamma.clone()),
                black_box(beta.clone()),
            )
        })
    });

    group.bench_function("rms_norm", |bench| {
        bench.iter(|| tensor_ops.rms_norm(black_box(input.clone()), black_box(gamma.clone())))
    });

    group.finish();
}

criterion_group!(
    benches,
    simd_dot_product_benchmark,
    vector_operations_benchmark,
    matrix_multiplication_benchmark,
    activation_functions_benchmark,
    attention_benchmark,
    layer_norm_benchmark
);
criterion_main!(benches);
