/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <array>
#include <stdexcept>
#include <string>

#include "ecf_codec.h"

namespace ECF {

static constexpr size_t kMaxBufferSize = 65535;

Decoder::Decoder() : num_events_(0) {}

size_t Decoder::operator()(const std::uint8_t *cur_raw_data, const std::uint8_t *raw_data_end, EventCD *event_ptr) {
    const std::uint32_t *info_ptr = reinterpret_cast<const std::uint32_t *>(cur_raw_data);
    num_events_                   = *info_ptr >> 2;
    ys_xs_and_ps_packed_          = (*info_ptr >> 1) & 1;
    xs_and_ps_packed_             = (*info_ptr >> 0) & 1;

    if (num_events_ > kMaxBufferSize) {
        throw std::runtime_error(std::string("Too many events to decode in buffer, maximum supported is ") +
                                 std::to_string(kMaxBufferSize));
    }
    ts_.resize(num_events_);
    // we allocate 5 extra values to not deal with special cases in last iteration of decode_ys_xs_and_ps_packed or
    // decode_xs_masked
    ys_.resize(num_events_ + 5);
    xs_.resize(num_events_ + 5);
    ps_.resize(num_events_ + 5);

    // skip 4 bytes header
    const std::uint8_t *ptr = cur_raw_data;
    ptr += 4;

    ptr += decode_ts(ptr);
    if (ys_xs_and_ps_packed_) {
        ptr += decode_ys_xs_and_ps_packed(ptr);
    } else {
        ptr += decode_ys(ptr);
        if (xs_and_ps_packed_) {
            ptr += decode_xs_and_ps_packed(ptr);
        } else {
            ptr += decode_xs_masked(ptr);
            ptr += decode_ps(ptr);
        }
    }

    transpose(event_ptr);
    return num_events_ * sizeof(EventCD);
}

size_t Decoder::getDecompressedSize(const std::uint32_t *header_ptr) const {
    auto num_events = *header_ptr >> 2;
    return num_events * sizeof(EventCD);
}

size_t Decoder::decode_ts(const std::uint8_t *ptr) {
    const std::uint64_t *orig_ptr = reinterpret_cast<const std::uint64_t *>(ptr);
    std::uint64_t t0              = *orig_ptr;
    ++orig_ptr;

    const std::uint8_t *ts_ptr = reinterpret_cast<const std::uint8_t *>(orig_ptr);
    std::uint64_t cur_t        = t0;
    for (size_t i = 0; i < num_events_;) {
        std::uint8_t v = *ts_ptr;
        std::uint8_t t = (v >> 4);
        size_t c       = v & 0b1111;

        if (t != 0b1111) {
            cur_t += t;
            if (c == 0b1111) {
                ++ts_ptr;
                c = *ts_ptr;
                ++ts_ptr;
                c = (*ts_ptr << 8) | c;
            }
            for (size_t j = 0; j < c; ++j) {
                ts_[i] = cur_t;
                ++i;
            }
            ++ts_ptr;
        } else {
            std::uint64_t dt = 0;
            for (int j = 0; t == 0b1111; ++j) {
                dt = dt | (c << (4 * j));
                ++ts_ptr;
                v = *ts_ptr;
                t = v >> 4;
                c = v & 0b1111;
            }
            cur_t += dt;
        }
    }

    return std::distance(ptr, reinterpret_cast<const std::uint8_t *>(ts_ptr));
}

size_t Decoder::decode_ys(const std::uint8_t *ptr) {
    const std::uint16_t *ys_ptr = reinterpret_cast<const std::uint16_t *>(ptr);
    for (size_t i = 0; i < num_events_;) {
        std::uint16_t v = *ys_ptr;
        std::uint16_t y = v >> 5, c = v & 0b11111;
        if (c == 0b11111) {
            ++ys_ptr;
            c = *ys_ptr;
        }
        for (size_t j = 0; j < c; ++j) {
            ys_[i] = y;
            ++i;
        }
        ++ys_ptr;
    }

    return std::distance(ptr, reinterpret_cast<const std::uint8_t *>(ys_ptr));
}

size_t Decoder::decode_ps(const std::uint8_t *ptr) {
    const std::uint8_t *ps_ptr = reinterpret_cast<const std::uint8_t *>(ptr);
    for (size_t i = 0; i < num_events_;) {
        std::uint8_t v = *ps_ptr;
        std::uint8_t p = v >> 7;
        size_t c       = v & 0b1111111;
        if (c == 0b1111111) {
            ++ps_ptr;
            c = *ps_ptr;
            ++ps_ptr;
            c = (*ps_ptr << 8) | c;
        }
        for (size_t j = 0; j < c; ++j) {
            ps_[i] = p;
            ++i;
        }
        ++ps_ptr;
    }

    return std::distance(ptr, reinterpret_cast<const std::uint8_t *>(ps_ptr));
}

size_t Decoder::decode_xs_masked(const std::uint8_t *ptr) {
    const std::uint16_t *xs_ptr = reinterpret_cast<const std::uint16_t *>(ptr);
    for (size_t i = 0; i < num_events_;) {
        std::uint16_t v = *xs_ptr;
        std::uint16_t x = v >> 5, m = v & 0b11111;
        xs_[i] = x;
        ++i;

        for (std::uint16_t j = 0; j < 5; ++j) {
            if ((m & (1 << (4 - j))) != 0) {
                xs_[i] = x + j + 1;
                ++i;
            }
        }
        ++xs_ptr;
    }

    return std::distance(ptr, reinterpret_cast<const std::uint8_t *>(xs_ptr));
}

size_t Decoder::decode_xs_and_ps_packed(const std::uint8_t *ptr) {
    const std::uint16_t *vs_ptr = reinterpret_cast<const std::uint16_t *>(ptr);
    for (size_t i = 0; i < num_events_; i += 4) {
        std::array<std::uint16_t, 4> vs;
        vs[0] = vs_ptr[0] >> 4;
        vs[1] = (vs_ptr[0] & 0b1111) | ((vs_ptr[1] >> 8) << 4);
        vs[2] = (vs_ptr[1] & 0b11111111) | ((vs_ptr[2] >> 12) << 8);
        vs[3] = (vs_ptr[2] & 0b111111111111);

        for (size_t j = 0; j < 4; ++j) {
            xs_[i + j] = vs[j] >> 1;
            ps_[i + j] = vs[j] & 1;
        }

        vs_ptr += 3;
    }

    return std::distance(ptr, reinterpret_cast<const std::uint8_t *>(vs_ptr));
}

size_t Decoder::decode_ys_xs_and_ps_packed(const std::uint8_t *ptr) {
    const std::uint32_t *vs_ptr = reinterpret_cast<const std::uint32_t *>(ptr);
    for (size_t i = 0; i < num_events_; i += 4) {
        std::array<std::uint32_t, 4> vs;
        vs[0] = (vs_ptr[0] >> 8);
        vs[1] = (vs_ptr[0] & 0xff) | ((vs_ptr[1] >> 16) << 8);
        vs[2] = (vs_ptr[1] & 0xffff) | ((vs_ptr[2] >> 24) << 16);
        vs[3] = (vs_ptr[2] & 0xffffff);

        for (size_t j = 0; j < 4; ++j) {
            ys_[i + j] = (vs[j] >> 12) & 0b11111111111;
            xs_[i + j] = (vs[j] >> 1) & 0b11111111111;
            ps_[i + j] = (vs[j] & 1);
        }

        vs_ptr += 3;
    }

    return std::distance(ptr, reinterpret_cast<const std::uint8_t *>(vs_ptr));
}

void Decoder::transpose(EventCD *ptr) {
    for (size_t i = 0; i < num_events_; ++i) {
        ptr[i].t = ts_[i];
        ptr[i].y = ys_[i];
        ptr[i].x = xs_[i];
        ptr[i].p = ps_[i];
    }
}

Encoder::Encoder() : count_(0) {
    ts_.resize(kMaxBufferSize);
    // we allocate 5 extra values to not deal with special cases in last iteration of encode_ys_xs_and_ps_packed or
    // encode_xs_masked their values are irrelevant (and thus, not reset in transpose), they will be ignored when
    // decoding, we just want to avoid out of bounds accesses
    ys_.resize(kMaxBufferSize + 5);
    xs_.resize(kMaxBufferSize + 5);
    ps_.resize(kMaxBufferSize + 5);
}

size_t Encoder::getCompressedSize() const {
    // gross overestimation ....
    return 4 + 8 + kMaxBufferSize * (4 + 2 + 1 + 2);
}

size_t Encoder::operator()(const EventCD *begin_ev_ptr, const EventCD *end_ev_ptr, std::uint8_t *raw_ev_ptr) {
    if (begin_ev_ptr > end_ev_ptr) {
        throw std::runtime_error(std::string("No events to encode, check the events range passed"));
    }
    if (static_cast<size_t>(std::distance(begin_ev_ptr, end_ev_ptr)) > kMaxBufferSize) {
        throw std::runtime_error(std::string("Too many events to encode in buffer, maximum allowed is ") +
                                 std::to_string(kMaxBufferSize));
    }
    transpose(begin_ev_ptr, end_ev_ptr);
    return encode(raw_ev_ptr);
}

void Encoder::transpose(const EventCD *begin, const EventCD *end) {
    count_ = 0;
    for (auto it = begin; it != end; ++it) {
        ts_[count_] = it->t;
        ys_[count_] = it->y;
        xs_[count_] = it->x;
        ps_[count_] = static_cast<std::uint8_t>(it->p);
        ++count_;
    }
}

size_t Encoder::encode(std::uint8_t *raw_ev_ptr) {
    auto ptr = raw_ev_ptr;

    // output # of events + ys/xs/ps encoding style
    double ratio            = 0.25;
    bool pack_ys_xs_and_ps  = should_pack_ys_xs_and_ps(ratio);
    bool pack_xs_and_ps     = !pack_ys_xs_and_ps && should_pack_xs_and_ps(ratio);
    std::uint32_t *info_ptr = reinterpret_cast<std::uint32_t *>(raw_ev_ptr);
    *info_ptr = static_cast<std::uint32_t>(count_ << 2) | static_cast<std::uint32_t>(pack_ys_xs_and_ps << 1) |
                static_cast<std::uint32_t>(pack_xs_and_ps);

    // skip 4 bytes header
    raw_ev_ptr += 4;

    // encode data
    raw_ev_ptr += encode_ts(raw_ev_ptr);
    if (pack_ys_xs_and_ps) {
        raw_ev_ptr += encode_ys_xs_and_ps_packed(raw_ev_ptr);
    } else {
        raw_ev_ptr += encode_ys(raw_ev_ptr);
        if (pack_xs_and_ps) {
            raw_ev_ptr += encode_xs_and_ps_packed(raw_ev_ptr);
        } else {
            raw_ev_ptr += encode_xs_masked(raw_ev_ptr);
            raw_ev_ptr += encode_ps(raw_ev_ptr);
        }
    }

    return std::distance(ptr, raw_ev_ptr);
}

bool Encoder::should_pack_ys_xs_and_ps(double ratio) const {
    size_t count                = 0;
    int prev_y                  = ys_[0];
    const size_t max_num_events = (count_ < (4 / ratio) ? count_ : static_cast<size_t>(ratio * count_ + 0.5));
    // empirically found value
    const size_t threshold = static_cast<size_t>(max_num_events * 20.0 / 100 + 0.5);

    for (size_t i = 1; i < max_num_events; ++i) {
        if (ys_[i] == prev_y) {
            ++count;
            if (count > threshold) {
                return false;
            }
        } else {
            prev_y = ys_[i];
        }
    }
    return true;
}

bool Encoder::should_pack_xs_and_ps(double ratio) const {
    size_t count = 0;
    int cur_x = xs_[0], prev_x = xs_[0];
    const size_t max_num_events = (count_ < (4 / ratio) ? count_ : static_cast<size_t>(ratio * count_ + 0.5));
    // empirically found value
    const size_t threshold = static_cast<size_t>(max_num_events * 8.0 / 100 + 0.5);

    for (size_t i = 1; i < max_num_events; ++i) {
        if (xs_[i] > prev_x && xs_[i] <= cur_x + 5) {
            ++count;
            if (count > threshold) {
                return false;
            }
        } else {
            cur_x = xs_[i];
        }
        prev_x = xs_[i];
    }
    return true;
}

size_t Encoder::encode_ts(std::uint8_t *ptr) {
    // output timestamp origin
    std::uint64_t *orig_ptr = reinterpret_cast<std::uint64_t *>(ptr);
    *orig_ptr               = ts_[0];
    ++orig_ptr;

    std::uint8_t *ts_ptr   = reinterpret_cast<std::uint8_t *>(orig_ptr);
    std::uint64_t cur_t    = ts_[0];
    std::uint8_t t         = 0;
    size_t c               = 1;
    const size_t max_count = (1 << 4) - 1;
    for (size_t i = 0; i < count_;) {
        if (ts_[i] >= cur_t + 0b1111) {
            std::uint64_t dt = ts_[i] - cur_t;
            while (dt > 0) {
                *ts_ptr = 0b11110000 | (dt & 0b1111);
                ++ts_ptr;
                dt >>= 4;
            }
            cur_t = ts_[i];
        }

        if (ts_[i] < cur_t) {
            // handle time discrepancies...
            t = 0;
        } else {
            t     = static_cast<std::uint8_t>(ts_[i] - cur_t);
            cur_t = ts_[i];
        }

        for (c = 1; c < count_ - i; ++c) {
            if (ts_[i + c] != cur_t) {
                break;
            }
        }

        if (c >= max_count) {
            *ts_ptr = (t << 4) | 0b1111;
            ++ts_ptr;
            *ts_ptr = c & 0xff;
            ++ts_ptr;
            *ts_ptr = (c >> 8) & 0xff;
            ++ts_ptr;
        } else {
            *ts_ptr = (t << 4) | static_cast<std::uint8_t>(c);
            ++ts_ptr;
        }

        i += c;
    }

    return std::distance(ptr, reinterpret_cast<std::uint8_t *>(ts_ptr));
}

size_t Encoder::encode_ys(std::uint8_t *ptr) {
    std::uint16_t *ys_ptr  = reinterpret_cast<std::uint16_t *>(ptr);
    unsigned short cur_y   = ys_[0];
    std::uint16_t y        = cur_y;
    size_t c               = 1;
    const size_t max_count = (1 << 5) - 1;
    for (size_t i = 0; i < count_;) {
        cur_y = ys_[i];
        y     = cur_y;

        for (c = 1; c < count_ - i; ++c) {
            if (ys_[i + c] != cur_y) {
                break;
            }
        }

        if (c >= max_count) {
            *ys_ptr = (y << 5) | 0b11111;
            ++ys_ptr;
            *ys_ptr = static_cast<std::uint16_t>(c);
            ++ys_ptr;
        } else {
            *ys_ptr = (y << 5) | static_cast<std::uint16_t>(c);
            ++ys_ptr;
        }

        i += c;
    }

    return std::distance(ptr, reinterpret_cast<std::uint8_t *>(ys_ptr));
}

size_t Encoder::encode_ps(std::uint8_t *ptr) {
    std::uint8_t *ps_ptr   = reinterpret_cast<std::uint8_t *>(ptr);
    short cur_p            = ps_[0];
    std::uint8_t p         = static_cast<std::uint8_t>(cur_p);
    size_t c               = 1;
    const size_t max_count = (1 << 7) - 1;
    for (size_t i = 0; i < count_;) {
        cur_p = ps_[i];
        p     = static_cast<std::uint8_t>(cur_p);

        for (c = 1; c < count_ - i; ++c) {
            if (ps_[i + c] != cur_p) {
                break;
            }
        }

        if (c >= max_count) {
            *ps_ptr = (p << 7) | 0b1111111;
            ++ps_ptr;
            *ps_ptr = c & 0xff;
            ++ps_ptr;
            *ps_ptr = (c >> 8) & 0xff;
            ++ps_ptr;
        } else {
            *ps_ptr = (p << 7) | static_cast<std::uint8_t>(c);
            ++ps_ptr;
        }

        i += c;
    }

    return std::distance(ptr, reinterpret_cast<std::uint8_t *>(ps_ptr));
}

size_t Encoder::encode_xs_masked(std::uint8_t *ptr) {
    std::uint16_t *xs_ptr = reinterpret_cast<std::uint16_t *>(ptr);
    for (size_t i = 0; i < count_;) {
        std::uint16_t x = xs_[i];
        ++i;

        *xs_ptr = x << 5;
        for (size_t j = 0; j < 5; ++j) {
            if (xs_[i] == x + j + 1) {
                *xs_ptr |= (1 << (4 - j));
                ++i;
            }
        }
        ++xs_ptr;
    }

    return std::distance(ptr, reinterpret_cast<std::uint8_t *>(xs_ptr));
}

size_t Encoder::encode_xs_and_ps_packed(std::uint8_t *ptr) {
    std::uint16_t *vs_ptr = reinterpret_cast<std::uint16_t *>(ptr);
    for (size_t i = 0; i < count_; i += 4) {
        std::array<std::uint16_t, 4> vs;
        for (size_t j = 0; j < 4; ++j) {
            vs[j] = (xs_[i + j] << 1) | ps_[i + j];
        }

        vs_ptr[0] = (vs[0] << 4) | (vs[1] & 0b1111);
        vs_ptr[1] = ((vs[1] >> 4) << 8) | (vs[2] & 0b11111111);
        vs_ptr[2] = ((vs[2] >> 8) << 12) | (vs[3] & 0b111111111111);

        vs_ptr += 3;
    }

    return std::distance(ptr, reinterpret_cast<std::uint8_t *>(vs_ptr));
}

size_t Encoder::encode_ys_xs_and_ps_packed(std::uint8_t *ptr) {
    std::uint32_t *vs_ptr = reinterpret_cast<std::uint32_t *>(ptr);
    for (size_t i = 0; i < count_; i += 4) {
        std::array<std::uint32_t, 4> vs;
        for (size_t j = 0; j < 4; ++j) {
            vs[j] = (ys_[i + j] << 12) | (xs_[i + j] << 1) | ps_[i + j];
        }

        vs_ptr[0] = (vs[0] << 8) | (vs[1] & 0xff);
        vs_ptr[1] = ((vs[1] >> 8) << 16) | (vs[2] & 0xffff);
        vs_ptr[2] = ((vs[2] >> 16) << 24) | (vs[3] & 0xffffff);

        vs_ptr += 3;
    }

    return std::distance(ptr, reinterpret_cast<std::uint8_t *>(vs_ptr));
}
} // namespace ECF
