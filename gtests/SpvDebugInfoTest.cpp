// Copyright (C) 2025 NVIDIA Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "TestFixture.h"
#include "glslang/Public/ResourceLimits.h"
#include <gtest/gtest.h>
#include <regex>
#include <sstream>
#include <string>

namespace glslangtest {

class SpvDebugInfoTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // Set up any common test state.
    }

    void TearDown() override
    {
        // Clean up any common test state.
    }

    // Helper function to compile shader with debug info and get SPIR-V disassembly.
    std::string compileShaderToSpirvWithDebugInfo(const std::string& shaderSource, EShLanguage stage, int spirvVersion = 130)
    {
        glslang::TShader shader(stage);
        glslang::TProgram program;

        // Enable debug info generation
        shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 110);
        shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
        shader.setEnvTarget(glslang::EShTargetSpv, static_cast<glslang::EShTargetLanguageVersion>(spirvVersion));

        // Compile the shader
        const char* shaderStrings = shaderSource.c_str();
        shader.setStrings(&shaderStrings, 1);

        EShMessages messages = (EShMessages)(EShMsgDefault | EShMsgDebugInfo);
        if (!shader.parse(GetDefaultResources(), 450, false, messages)) {
            return "COMPILATION_FAILED: " + std::string(shader.getInfoLog());
        }

        program.addShader(&shader);
        if (!program.link(messages)) {
            return "LINKING_FAILED: " + std::string(program.getInfoLog());
        }

        // Generate SPIR-V with debug info.
        glslang::SpvOptions spvOptions;
        spvOptions.generateDebugInfo = true;
        spvOptions.emitNonSemanticShaderDebugInfo = true;
        spvOptions.disableOptimizer = true;
        spvOptions.optimizeSize = false;

        std::vector<uint32_t> spirv;
        glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &spvOptions);

        // Disassemble SPIR-V to text.
        std::ostringstream disassembly_stream;
        spv::Disassemble(disassembly_stream, spirv);

        return disassembly_stream.str();
    }

    // Helper function to check if the given SPIR-V string contains a specific pattern.
    bool containsPattern(const std::string& spirvText, const std::string& pattern)
    {
        return spirvText.find(pattern) != std::string::npos;
    }

    // Helper to check if debug info extension is imported
    bool containsDebugInfoExtension(const std::string& spirvText)
    {
        return containsPattern(spirvText, "ExtInstImport  \"NonSemantic.Shader.DebugInfo.100\"");
    }

    // Helper to check for DebugTypeCooperativeVectorNV (instruction 109)
    bool containsDebugTypeCooperativeVectorNV(const std::string& spirvText)
    {
        // Look for ExtInst with opcode 109
        return containsPattern(spirvText, "ExtInst 1(NonSemantic.Shader.DebugInfo.100) 109");
    }

    // Helper to check for DebugTypeCooperativeMatrixNV (instruction 110)
    bool containsDebugTypeCooperativeMatrixNV(const std::string& spirvText)
    {
        // Look for ExtInst with opcode 110
        return containsPattern(spirvText, "ExtInst 1(NonSemantic.Shader.DebugInfo.100) 110");
    }

    // Helper to check for DebugTypeCooperativeMatrixKHR (instruction 111)
    bool containsDebugTypeCooperativeMatrixKHR(const std::string& spirvText)
    {
        // Look for ExtInst with opcode 111
        return containsPattern(spirvText, "ExtInst 1(NonSemantic.Shader.DebugInfo.100) 111");
    }

    // Helper to check for TypeCooperativeVectorNV
    bool containsTypeCooperativeVectorNV(const std::string& spirvText)
    {
        return containsPattern(spirvText, "TypeCooperativeVectorNV");
    }

    // Helper to check for TypeCooperativeMatrixNV
    bool containsTypeCooperativeMatrixNV(const std::string& spirvText)
    {
        return containsPattern(spirvText, "TypeCooperativeMatrixNV");
    }

    // Helper to check for TypeCooperativeMatrixKHR
    bool containsTypeCooperativeMatrixKHR(const std::string& spirvText)
    {
        return containsPattern(spirvText, "TypeCooperativeMatrixKHR");
    }
};

// Test 1: Cooperative vector with regular constants should generate DebugTypeCooperativeVectorNV
TEST_F(SpvDebugInfoTest, CooperativeVectorGeneratesDebugType)
{
    const std::string shaderSource = R"(
        #version 450 core
        #extension GL_KHR_memory_scope_semantics : enable
        #extension GL_NV_cooperative_vector : enable
        #extension GL_EXT_shader_explicit_arithmetic_types : enable

        layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer MatrixBuf {
            float16_t data[];
        } matrixBuf;

        void main()
        {
            // Use regular constants (not specialization constants)
            coopvecNV<float, 16> vec16 = coopvecNV<float, 16>(1.0);
            coopvecNV<float, 8> vec8 = coopvecNV<float, 8>(2.0);
            coopvecNV<float16_t, 4> vec4_f16 = coopvecNV<float16_t, 4>(float16_t(3.0));

            vec16 = vec16 + vec16;
            vec8 = vec8 * 2.0;
            vec4_f16 = vec4_f16 - vec4_f16;
        }
    )";

    std::string spirv = compileShaderToSpirvWithDebugInfo(shaderSource, EShLangCompute);

    // Verify debug info extension is imported
    EXPECT_TRUE(containsDebugInfoExtension(spirv))
        << "SPIR-V should contain NonSemantic.Shader.DebugInfo.100 extension.\n"
        << "Generated SPIR-V:\n"
        << spirv;

    // Verify cooperative vector types are generated
    EXPECT_TRUE(containsTypeCooperativeVectorNV(spirv))
        << "SPIR-V should contain TypeCooperativeVectorNV.\n"
        << "Generated SPIR-V:\n"
        << spirv;

    // Verify debug types for cooperative vectors are generated (instruction 109)
    EXPECT_TRUE(containsDebugTypeCooperativeVectorNV(spirv))
        << "SPIR-V should contain DebugTypeCooperativeVectorNV (instruction 109) for cooperative vector types.\n"
        << "Generated SPIR-V:\n"
        << spirv;
}

// Test 2: Cooperative vector with specialization constants should generate debug types
TEST_F(SpvDebugInfoTest, CooperativeVectorWithSpecConstantsGeneratesDebugType)
{
    const std::string shaderSource = R"(
        #version 450 core
        #extension GL_KHR_memory_scope_semantics : enable
        #extension GL_NV_cooperative_vector : enable
        #extension GL_EXT_shader_explicit_arithmetic_types : enable

        layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(constant_id = 0) const uint SIZE = 16;

        void main()
        {
            // Use specialization constant for size
            coopvecNV<float, SIZE> vec = coopvecNV<float, SIZE>(1.0);
        }
    )";

    std::string spirv = compileShaderToSpirvWithDebugInfo(shaderSource, EShLangCompute);

    // Verify debug info extension is imported
    EXPECT_TRUE(containsDebugInfoExtension(spirv))
        << "SPIR-V should contain NonSemantic.Shader.DebugInfo.100 extension.\n";

    // Cooperative vector types should be generated
    EXPECT_TRUE(containsTypeCooperativeVectorNV(spirv))
        << "SPIR-V should contain TypeCooperativeVectorNV.\n";

    // Debug types should be generated even with specialization constants
    // The spec allows any "constant instruction" including OpSpecConstant
    EXPECT_TRUE(containsDebugTypeCooperativeVectorNV(spirv))
        << "SPIR-V should contain DebugTypeCooperativeVectorNV (instruction 109) even when using specialization constants.\n"
        << "Generated SPIR-V:\n"
        << spirv;
}

// Test 3: Cooperative matrix NV with regular constants should generate DebugTypeCooperativeMatrixNV
TEST_F(SpvDebugInfoTest, CooperativeMatrixNVGeneratesDebugType)
{
    const std::string shaderSource = R"(
        #version 450 core
        #extension GL_KHR_memory_scope_semantics : enable
        #extension GL_NV_cooperative_matrix : enable
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

        layout (local_size_x = 32) in;

        layout(set = 0, binding = 0) coherent buffer Block {
            float16_t data[];
        } block;

        void main()
        {
            // Use regular constants
            fcoopmatNV<16, gl_ScopeSubgroup, 16, 8> mat = fcoopmatNV<16, gl_ScopeSubgroup, 16, 8>(0.0);
            mat = mat + mat;
            coopMatStoreNV(mat, block.data, 0, 16, false);
        }
    )";

    std::string spirv = compileShaderToSpirvWithDebugInfo(shaderSource, EShLangCompute);

    // Verify cooperative matrix types are generated
    EXPECT_TRUE(containsTypeCooperativeMatrixNV(spirv))
        << "SPIR-V should contain TypeCooperativeMatrixNV.\n";

    // Verify debug types for cooperative matrices are generated (instruction 110)
    EXPECT_TRUE(containsDebugTypeCooperativeMatrixNV(spirv))
        << "SPIR-V should contain DebugTypeCooperativeMatrixNV (instruction 110) for cooperative matrix types.\n"
        << "Generated SPIR-V:\n"
        << spirv;
}

// Test 4: Cooperative matrix KHR with regular constants should generate DebugTypeCooperativeMatrixKHR
TEST_F(SpvDebugInfoTest, CooperativeMatrixKHRGeneratesDebugType)
{
    const std::string shaderSource = R"(
        #version 450 core
        #pragma use_vulkan_memory_model
        #extension GL_KHR_memory_scope_semantics : enable
        #extension GL_KHR_cooperative_matrix : enable
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

        layout (local_size_x = 32) in;

        layout(set = 0, binding = 0) buffer Block {
            float16_t data[];
        } block;

        void main()
        {
            // Use regular constants
            coopmat<float16_t, gl_ScopeSubgroup, 16, 8, gl_MatrixUseA> matA;
            coopmat<float16_t, gl_ScopeSubgroup, 8, 16, gl_MatrixUseB> matB;
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;

            matC = coopMatMulAdd(matA, matB, matC);
        }
    )";

    std::string spirv = compileShaderToSpirvWithDebugInfo(shaderSource, EShLangCompute, 110);

    // Verify cooperative matrix types are generated
    EXPECT_TRUE(containsTypeCooperativeMatrixKHR(spirv))
        << "SPIR-V should contain TypeCooperativeMatrixKHR.\n";

    // Verify debug types for cooperative matrices are generated (instruction 111)
    EXPECT_TRUE(containsDebugTypeCooperativeMatrixKHR(spirv))
        << "SPIR-V should contain DebugTypeCooperativeMatrixKHR (instruction 111) for KHR cooperative matrix types.\n"
        << "Generated SPIR-V:\n"
        << spirv;
}

// Test 5: Cooperative matrix KHR with specialization constants should generate debug types
TEST_F(SpvDebugInfoTest, CooperativeMatrixKHRWithSpecConstantsGeneratesDebugType)
{
    const std::string shaderSource = R"(
        #version 450 core
        #pragma use_vulkan_memory_model
        #extension GL_KHR_memory_scope_semantics : enable
        #extension GL_KHR_cooperative_matrix : enable
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

        layout (local_size_x = 32) in;

        layout(constant_id = 0) const uint M = 16;
        layout(constant_id = 1) const uint N = 8;

        layout(set = 0, binding = 0) buffer Block {
            float16_t data[];
        } block;

        void main()
        {
            // Use specialization constants for dimensions
            coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseA> mat =
                coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseA>(float16_t(0.0));
            // Actually use the variable to prevent optimization
            coopMatStore(mat, block.data, 0, 0, gl_CooperativeMatrixLayoutRowMajor);
        }
    )";

    std::string spirv = compileShaderToSpirvWithDebugInfo(shaderSource, EShLangCompute, 110);

    // Cooperative matrix types should be generated
    EXPECT_TRUE(containsTypeCooperativeMatrixKHR(spirv))
        << "SPIR-V should contain TypeCooperativeMatrixKHR.\n"
        << "Generated SPIR-V:\n"
        << spirv;

    // Debug types should be generated even with specialization constants
    // The spec allows any "constant instruction" including OpSpecConstant
    EXPECT_TRUE(containsDebugTypeCooperativeMatrixKHR(spirv))
        << "SPIR-V should contain DebugTypeCooperativeMatrixKHR (instruction 111) even when using specialization constants.\n"
        << "Generated SPIR-V:\n"
        << spirv;
}

} // namespace glslangtest
