#include <functional>

class Autograd {
  public:
    void push(std::function<void()> backward);
    void backward();
    [[nodiscard]] bool empty() const;

  private:
    std::vector<std::function<void()>> nodes_;
};

Autograd &autograd();