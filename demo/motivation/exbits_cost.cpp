#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace costmodel {

struct Weights {
	// Integer/SIMD (FastScan inner loop)
	double w_load64 = 1.0;     // 64B load (AVX512 path)
	double w_load32 = 1.0;     // 32B load (AVX2 path)
	double w_and = 1.0;        // integer bitwise AND
	double w_srli16 = 1.0;     // shift right logical 16-bit lanes
	double w_shuffle = 1.0;    // byte shuffle (epi8)
	double w_add16 = 1.0;      // 16-bit integer add
	// Float ops
	double w_fma = 1.0;        // fused multiply-add
	double w_fmul = 1.0;       // float mul
	double w_fadd = 1.0;       // float add
	// Single-vector 1-bit path (mask load + vector add)
	double w_maskload = 1.0;   // masked vector load (conceptual)
	double w_vaddps = 1.0;     // vector add ps
};

struct Counts {
	// FastScan integer-side ops (aggregated over all iterations)
	uint64_t load64 = 0;
	uint64_t load32 = 0;
	uint64_t iand = 0;
	uint64_t srli16 = 0;
	uint64_t shuffle = 0;
	uint64_t add16 = 0;
	uint64_t merge_const = 0; // optional constant merge cost per batch
	// Float ops
	uint64_t fma = 0;
	uint64_t fmul = 0;
	uint64_t fadd = 0;
	// Single-vector 1-bit ops
	uint64_t maskload = 0;
	uint64_t vaddps = 0;

	double weighted_total(const Weights& w) const {
		double s = 0.0;
		s += load64 * w.w_load64;
		s += load32 * w.w_load32;
		s += iand * w.w_and;
		s += srli16 * w.w_srli16;
		s += shuffle * w.w_shuffle;
		s += add16 * w.w_add16;
		s += merge_const; // treat as already-weighted units
		s += fma * w.w_fma;
		s += fmul * w.w_fmul;
		s += fadd * w.w_fadd;
		s += maskload * w.w_maskload;
		s += vaddps * w.w_vaddps;
		return s;
	}
};

static inline uint64_t ceil_div_uint64(uint64_t a, uint64_t b) {
	return (a + b - 1) / b;
}

static inline uint64_t round_up_uint64(uint64_t x, uint64_t align) {
	if (align == 0) return x;
	return ((x + align - 1) / align) * align;
}

struct Inputs {
	uint64_t N = 0;           // number of vectors to scan (bucket) or database size used by graph path-length estimation
	uint64_t dim = 0;         // original dimension
	uint64_t bits = 0;        // total bits per dim (B); ex_bits = B-1
	uint64_t degree = 0;      // graph average degree (optional, used for lavg estimation if requested)
	double lavg = -1.0;        // average path length (required for exact graph cost); if negative and estimate flag used, we estimate
	uint64_t align = 16;      // padded alignment step (default 16)
	std::string arch = "avx512"; // or "avx2"
	bool json = false;        // output JSON instead of text
	double merge_cost_per_batch = 0.0; // optional constant merge cost per batch (FastScan)
	bool estimate_lavg = false;        // whether to estimate lavg from N, degree
};

static void print_usage(const char* prog) {
	std::cerr
		<< "Usage: " << prog << " --N <num> --dim <dim> --bits <B> [--degree <deg>] [--lavg <L>]\n"
		<< "             [--arch avx512|avx2] [--align <k>] [--json]\\n"
		<< "             [--merge <const>] [--estimate-lavg]\\n"
		<< "             [--w-load64 x] [--w-load32 x] [--w-and x] [--w-srli16 x] [--w-shuffle x] [--w-add16 x]\\n"
		<< "             [--w-fma x] [--w-fmul x] [--w-fadd x] [--w-maskload x] [--w-vaddps x]\n\n"
		<< "Notes:\n"
		<< "- For exact graph cost, pass --lavg explicitly. If --estimate-lavg is set and --lavg is omitted,\n"
		<< "  we estimate lavg approximately as max(1, ln(N)/ln(max(2,degree))).\n"
		<< "- Costs are reported as operation counts and as a weighted total cost. Adjust weights to your platform.\n"
		<< std::endl;
}

static bool parse_args(int argc, char** argv, Inputs& in, Weights& w) {
	std::map<std::string, std::string> kv;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.rfind("--", 0) == 0) {
			std::string key = a;
			std::string val;
			if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
				val = argv[++i];
			}
			kv[key] = val;
		}
	}
	auto get_u64 = [&](const std::string& k, uint64_t defv) {
		auto it = kv.find(k);
		if (it == kv.end() || it->second.empty()) return defv;
		return static_cast<uint64_t>(std::strtoull(it->second.c_str(), nullptr, 10));
	};
	auto get_dbl = [&](const std::string& k, double defv) {
		auto it = kv.find(k);
		if (it == kv.end() || it->second.empty()) return defv;
		return std::strtod(it->second.c_str(), nullptr);
	};
	auto get_str = [&](const std::string& k, const std::string& defv) {
		auto it = kv.find(k);
		if (it == kv.end() || it->second.empty()) return defv;
		return it->second;
	};
	auto has = [&](const std::string& k) { return kv.find(k) != kv.end(); };

	in.N = get_u64("--N", in.N);
	in.dim = get_u64("--dim", in.dim);
	in.bits = get_u64("--bits", in.bits);
	in.degree = get_u64("--degree", in.degree);
	in.align = get_u64("--align", in.align);
	in.arch = get_str("--arch", in.arch);
	in.json = has("--json");
	in.merge_cost_per_batch = get_dbl("--merge", in.merge_cost_per_batch);
	in.estimate_lavg = has("--estimate-lavg");
	{
		auto it = kv.find("--lavg");
		if (it != kv.end() && !it->second.empty()) in.lavg = std::strtod(it->second.c_str(), nullptr);
	}

	// Weights
	w.w_load64 = get_dbl("--w-load64", w.w_load64);
	w.w_load32 = get_dbl("--w-load32", w.w_load32);
	w.w_and = get_dbl("--w-and", w.w_and);
	w.w_srli16 = get_dbl("--w-srli16", w.w_srli16);
	w.w_shuffle = get_dbl("--w-shuffle", w.w_shuffle);
	w.w_add16 = get_dbl("--w-add16", w.w_add16);
	w.w_fma = get_dbl("--w-fma", w.w_fma);
	w.w_fmul = get_dbl("--w-fmul", w.w_fmul);
	w.w_fadd = get_dbl("--w-fadd", w.w_fadd);
	w.w_maskload = get_dbl("--w-maskload", w.w_maskload);
	w.w_vaddps = get_dbl("--w-vaddps", w.w_vaddps);

	if (in.N == 0 || in.dim == 0 || in.bits == 0) return false;
	if (in.arch != "avx512" && in.arch != "avx2") return false;
	return true;
}

struct Result {
	Counts bucket;
	Counts graph;
	uint64_t Dp = 0;
	double lavg_used = 0.0;
};

static Result evaluate(const Inputs& in) {
	Result r{};
	const uint64_t Dp = round_up_uint64(in.dim, in.align);
	r.Dp = Dp;
	const uint64_t ex_bits = (in.bits > 0 ? (in.bits - 1) : 0);

	// -------------- Bucket Scan -----------------
	// Nb batches of FastScan; each batch iterates (Dp/16) times
	const uint64_t Nb = ceil_div_uint64(in.N, 32);
	const uint64_t iter = (Dp / 16);
	if (in.arch == "avx512") {
		r.bucket.load64 += Nb * (2ULL * iter);
		r.bucket.iand += Nb * (2ULL * iter);
		r.bucket.srli16 += Nb * (1ULL * iter);
		r.bucket.shuffle += Nb * (2ULL * iter);
		r.bucket.add16 += Nb * (4ULL * iter);
	} else {
		// AVX2 path: 4x32B loads per iter, rest same counts as spec
		r.bucket.load32 += Nb * (4ULL * iter);
		r.bucket.iand += Nb * (2ULL * iter);
		r.bucket.srli16 += Nb * (2ULL * iter);
		r.bucket.shuffle += Nb * (2ULL * iter);
		r.bucket.add16 += Nb * (4ULL * iter);
	}
	// optional per-batch constant merge cost (user-configurable)
	if (in.merge_cost_per_batch > 0.0) {
		r.bucket.merge_const += static_cast<uint64_t>(std::llround(in.merge_cost_per_batch * static_cast<double>(Nb)));
	}
	// Per vector scalar floats for ex-bits path
	// ip_est: 1 mul + 1 add, ex-dist combine: 2 mul + 4 add => total 3 mul + 5 add per vector
	r.bucket.fmul += in.N * 3ULL;
	r.bucket.fadd += in.N * 5ULL;
	// ex-code inner product: for ex_bits in {1..7} except 8, FMA count = Dp/16; for 8 bits, D mul + (D-1) add
	if (ex_bits == 8) {
		// rare, but support
		r.bucket.fmul += in.N * Dp;
		r.bucket.fadd += in.N * (Dp > 0 ? (Dp - 1) : 0);
	} else if (ex_bits >= 1) {
		r.bucket.fma += in.N * (Dp / 16);
	}

	// -------------- Graph Search ----------------
	double lavg_used = in.lavg;
	if (lavg_used <= 0.0) {
		if (in.estimate_lavg) {
			double deg = static_cast<double>(in.degree);
			if (deg < 2.0) deg = 2.0;
			double vN = static_cast<double>(in.N > 0 ? in.N : 1);
			lavg_used = std::max(1.0, std::log(vN) / std::log(deg));
		} else {
			// if no lavg, set to 0 to indicate unavailable
			lavg_used = 0.0;
		}
	}
	r.lavg_used = lavg_used;
	const uint64_t Lsteps = static_cast<uint64_t>(std::llround(lavg_used));
	if (Lsteps > 0) {
		// Single-vector 1-bit: mask_ip_x0_q style — per 64 dims block: 4 masked loads + 4 vector adds
		const uint64_t blocks64 = (Dp / 64);
		r.graph.maskload += Lsteps * (4ULL * blocks64);
		r.graph.vaddps += Lsteps * (4ULL * blocks64);
		// ex-bits inner product per node
		if (ex_bits == 8) {
			r.graph.fmul += Lsteps * Dp;
			r.graph.fadd += Lsteps * (Dp > 0 ? (Dp - 1) : 0);
		} else if (ex_bits >= 1) {
			r.graph.fma += Lsteps * (Dp / 16);
		}
		// final combine: 2 mul + 4 add per node
		r.graph.fmul += Lsteps * 2ULL;
		r.graph.fadd += Lsteps * 4ULL;
	}

	return r;
}

static void print_text(const Inputs& in, const Weights& w, const Result& r) {
	std::cout << "Inputs:\n";
	std::cout << "  N=" << in.N << ", dim=" << in.dim << ", padded_dim=" << r.Dp
			  << ", bits=" << in.bits << " (ex_bits=" << (in.bits ? (in.bits - 1) : 0) << ")"
			  << ", arch=" << in.arch << ", align=" << in.align << "\n";
	if (r.lavg_used > 0.0) {
		std::cout << "  lavg=" << r.lavg_used << " (graph steps)\n";
	} else {
		std::cout << "  lavg not provided (graph cost omitted).\n";
	}
	if (in.merge_cost_per_batch > 0.0) {
		std::cout << "  merge_cost_per_batch=" << in.merge_cost_per_batch << " (added to bucket)\n";
	}
	std::cout << "\nBucket Scan (ex-bits) counts:\n";
	std::cout << "  load64=" << r.bucket.load64 << ", load32=" << r.bucket.load32
			  << ", and=" << r.bucket.iand << ", srli16=" << r.bucket.srli16
			  << ", shuffle=" << r.bucket.shuffle << ", add16=" << r.bucket.add16
			  << ", fma=" << r.bucket.fma << ", fmul=" << r.bucket.fmul << ", fadd=" << r.bucket.fadd
			  << ", maskload=" << r.bucket.maskload << ", vaddps=" << r.bucket.vaddps
			  << ", merge_const=" << r.bucket.merge_const << "\n";
	double bucket_total = r.bucket.weighted_total(w);
	std::cout << "  Weighted total: " << std::fixed << std::setprecision(2) << bucket_total << "\n\n";

	std::cout << "Graph Search (ex-bits) counts:" << (r.lavg_used > 0.0 ? "" : " (lavg missing)") << "\n";
	std::cout << "  load64=" << r.graph.load64 << ", load32=" << r.graph.load32
			  << ", and=" << r.graph.iand << ", srli16=" << r.graph.srli16
			  << ", shuffle=" << r.graph.shuffle << ", add16=" << r.graph.add16
			  << ", fma=" << r.graph.fma << ", fmul=" << r.graph.fmul << ", fadd=" << r.graph.fadd
			  << ", maskload=" << r.graph.maskload << ", vaddps=" << r.graph.vaddps
			  << ", merge_const=" << r.graph.merge_const << "\n";
	double graph_total = r.graph.weighted_total(w);
	std::cout << "  Weighted total: " << std::fixed << std::setprecision(2) << graph_total << "\n";
}

static void print_json(const Inputs& in, const Weights& w, const Result& r) {
	auto q = [&](uint64_t v) { return std::to_string(v); };
	auto qd = [&](double v) { std::ostringstream os; os << std::fixed << std::setprecision(6) << v; return os.str(); };
	std::cout << "{\n";
	std::cout << "  \"inputs\": {\n";
	std::cout << "    \"N\": " << in.N << ", \"dim\": " << in.dim << ", \"padded_dim\": " << r.Dp << ", \"bits\": " << in.bits << ", \"ex_bits\": " << (in.bits ? (in.bits - 1) : 0) << ",\n";
	std::cout << "    \"arch\": \"" << in.arch << "\", \"align\": " << in.align << ", \"lavg\": " << qd(r.lavg_used) << ", \"merge_cost_per_batch\": " << qd(in.merge_cost_per_batch) << "\n";
	std::cout << "  },\n";
	std::cout << "  \"bucket\": {\n";
	std::cout << "    \"load64\": " << q(r.bucket.load64) << ", \"load32\": " << q(r.bucket.load32) << ", \"and\": " << q(r.bucket.iand) << ", \"srli16\": " << q(r.bucket.srli16) << ", \"shuffle\": " << q(r.bucket.shuffle) << ", \"add16\": " << q(r.bucket.add16) << ",\n";
	std::cout << "    \"fma\": " << q(r.bucket.fma) << ", \"fmul\": " << q(r.bucket.fmul) << ", \"fadd\": " << q(r.bucket.fadd) << ", \"maskload\": " << q(r.bucket.maskload) << ", \"vaddps\": " << q(r.bucket.vaddps) << ", \"merge_const\": " << q(r.bucket.merge_const) << ",\n";
	std::cout << "    \"weighted_total\": " << qd(r.bucket.weighted_total(w)) << "\n";
	std::cout << "  },\n";
	std::cout << "  \"graph\": {\n";
	std::cout << "    \"load64\": " << q(r.graph.load64) << ", \"load32\": " << q(r.graph.load32) << ", \"and\": " << q(r.graph.iand) << ", \"srli16\": " << q(r.graph.srli16) << ", \"shuffle\": " << q(r.graph.shuffle) << ", \"add16\": " << q(r.graph.add16) << ",\n";
	std::cout << "    \"fma\": " << q(r.graph.fma) << ", \"fmul\": " << q(r.graph.fmul) << ", \"fadd\": " << q(r.graph.fadd) << ", \"maskload\": " << q(r.graph.maskload) << ", \"vaddps\": " << q(r.graph.vaddps) << ", \"merge_const\": " << q(r.graph.merge_const) << ",\n";
	std::cout << "    \"weighted_total\": " << qd(r.graph.weighted_total(w)) << "\n";
	std::cout << "  }\n";
	std::cout << "}\n";
}

} // namespace costmodel

int main(int argc, char** argv) {
	using namespace costmodel;
	Inputs in;
	Weights w;
	if (!parse_args(argc, argv, in, w)) {
		print_usage(argv[0]);
		return 1;
	}
	Result r = evaluate(in);
	if (in.json) print_json(in, w, r); else print_text(in, w, r);
	return 0;
} 