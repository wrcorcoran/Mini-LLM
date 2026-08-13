#include "autograd.h"

Autograd &autograd() {
    static Autograd instance;
    return instance;
}

void Autograd::push(std::function<void()> backward) { nodes_.push_back(std::move(backward)); }

bool Autograd::empty() const { return nodes_.empty(); }

void Autograd::backward() {
    while (!empty()) {
        std::function<void()> top = std::move(nodes_.back());
        nodes_.pop_back();
        top();
    }
}