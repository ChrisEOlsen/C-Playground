#include <atomic>
#include <chrono>
#include <immintrin.h> // AVX2 Intrinsics
#include <iostream>
#include <thread>
#include <vector>

// AVX2 Implementation of the Monte Carlo Pi Approximation
void calculate_hits(int iterations, std::atomic<long long> &total_hits) {
  long long local_hits = 0;
  // Initial seeds for 8 parallel RNG states
  // We use the address of a local variable + offsets to ensure unique seeds per thread
  uint32_t seed_base = (uint32_t)(uintptr_t)&local_hits;

  // Initialize 8 separate 32-bit integers into a single 256-bit register
  // This vector represents the "current state" for 8 independent RNGs
  __m256i state = _mm256_setr_epi32(
      seed_base, seed_base + 1, seed_base + 2, seed_base + 3,
      seed_base + 4, seed_base + 5, seed_base + 6, seed_base + 7);

  // Constants for Xorshift32 algorithm and Float conversion
  const __m256i shift1 = _mm256_set1_epi32(13);
  const __m256i shift2 = _mm256_set1_epi32(17);
  const __m256i shift3 = _mm256_set1_epi32(5);

  // Constants for IEEE 754 float conversion trick
  // We want to convert a random 32-bit int to a float in [0, 1)
  // Trick: Construct a float in [1.0, 2.0) using bitwise OR, then subtract 1.0
  const __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
  const __m256i one_point_zero_bits = _mm256_set1_epi32(0x3F800000);
  const __m256 one_point_zero = _mm256_set1_ps(1.0f);

  // Process 8 iterations at a time
  for (int i = 0; i < iterations; i += 8) {
    
    // --- Generate X (8 values) ---
    __m256i x_int = state;
    x_int = _mm256_xor_si256(x_int, _mm256_slli_epi32(x_int, 13));
    x_int = _mm256_xor_si256(x_int, _mm256_srli_epi32(x_int, 17));
    x_int = _mm256_xor_si256(x_int, _mm256_slli_epi32(x_int, 5));
    state = x_int; // Update state

    // Convert to Float [0, 1)
    __m256i x_masked = _mm256_and_si256(x_int, mantissa_mask);
    __m256i x_bits = _mm256_or_si256(x_masked, one_point_zero_bits);
    __m256 x_vec = _mm256_castsi256_ps(x_bits); // Reinterpret bits as float
    x_vec = _mm256_sub_ps(x_vec, one_point_zero);

    // --- Generate Y (8 values) ---
    // We run the RNG step again to get Y
    __m256i y_int = state;
    y_int = _mm256_xor_si256(y_int, _mm256_slli_epi32(y_int, 13));
    y_int = _mm256_xor_si256(y_int, _mm256_srli_epi32(y_int, 17));
    y_int = _mm256_xor_si256(y_int, _mm256_slli_epi32(y_int, 5));
    state = y_int; // Update state again

    // Convert to Float [0, 1)
    __m256i y_masked = _mm256_and_si256(y_int, mantissa_mask);
    __m256i y_bits = _mm256_or_si256(y_masked, one_point_zero_bits);
    __m256 y_vec = _mm256_castsi256_ps(y_bits);
    y_vec = _mm256_sub_ps(y_vec, one_point_zero);

    // --- Calculate Distance Squared (x*x + y*y) ---
    __m256 x_sq = _mm256_mul_ps(x_vec, x_vec);
    __m256 y_sq = _mm256_mul_ps(y_vec, y_vec);
    __m256 dist_sq = _mm256_add_ps(x_sq, y_sq);

    // --- Check hits (dist_sq <= 1.0) ---
    // cmp_ps returns 0xFFFFFFFF for true, 0x00000000 for false
    __m256 cmp = _mm256_cmp_ps(dist_sq, one_point_zero, _CMP_LE_OQ);

    // --- Count Hits ---
    // movemask takes the most significant bit of each float lane and packs them into an int (8 bits)
    int mask = _mm256_movemask_ps(cmp);
    
    // popcnt counts the number of set bits (1s) in the integer
    local_hits += _mm_popcnt_u32(mask);
  }

  total_hits += local_hits;
}

int main() {
  const int TOTAL_ITERATIONS = 2'000'000'000; // 1 billion
  const int THREAD_COUNT = std::thread::hardware_concurrency();

  std::cout << "Launching " << TOTAL_ITERATIONS << " iterations across "
            << THREAD_COUNT << " threads (using AVX2)..." << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  std::atomic<long long> total_hits{0};
  int iterations_per_thread = TOTAL_ITERATIONS / THREAD_COUNT;

  // Launch threads
  for (int i = 0; i < THREAD_COUNT; ++i) {
    threads.emplace_back(calculate_hits, iterations_per_thread,
                         std::ref(total_hits));
  }

  // Join threads
  for (auto &t : threads) {
    t.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;

  double pi_estimate = 4.0 * total_hits / TOTAL_ITERATIONS;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Estimated Pi: " << pi_estimate << std::endl;
  std::cout << "Time Taken:   " << elapsed.count() << " seconds" << std::endl;
  std::cout << "Calculations per second: "
            << (TOTAL_ITERATIONS / elapsed.count()) / 1e6 << " Million"
            << std::endl;

  return 0;
}
