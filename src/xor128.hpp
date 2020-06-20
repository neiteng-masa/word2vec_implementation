
#ifndef XOR128_HPP
#define XOR128_HPP 

class xor128 {
    public:
        xor128() noexcept : x(123456789), y(362436069), z(521288629), w(88675123) {}
        double operator()() noexcept {
            uint32_t t;
            t = x ^ (x << 11);
            x = y; y = z; z = w;
            w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
            return w / (double)0x100000000;
        }
    private:
        uint32_t x, y, z, w;
};

#endif
