CXXFLAGS := $(pkg-config --libs --cflags optima-v2-runtime)

BUILD_DIR ?= build-make
OUTPUT_NAME := optimium-cpp
SOURCES := main.cpp
OUTPUTS := $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/$(OUTPUT_NAME): $(BUILD_DIR) $(OUTPUTS)
	$(CXX) -o $@ $(OUTPUTS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $<

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(OUTPUTS:%.o=%.d)