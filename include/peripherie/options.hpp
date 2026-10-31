#pragma once

enum class Scaling {
    STRONG,
    WEAK
};

inline const char* scaling_name(Scaling scaling) {
    switch (scaling) {
        case Scaling::STRONG: return "strong";
        case Scaling::WEAK:   return "weak";
    }
    return "unknown";
}