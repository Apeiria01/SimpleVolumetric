

#include <windows.h>

#include "d3dx12.h"
#include <string>
#include <wrl.h>
#include <cuda_runtime.h>
#include "ShaderStructs.h"
#include "CUDARasterizerLoop.h"
#include <dxgi1_5.h>
#include "WindowsSecurityAttributes.h"


using Device::FrameCount;

CUDARasterizerLoop::CUDARasterizerLoop(UINT width, UINT height, std::string name)
    : DX12WindowLoop(width, height, name),
    m_frameIndex(0),
    m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
    m_fenceValues{},
    m_rtvDescriptorSize(0) {
    m_viewport = { 0.0f, 0.0f, static_cast<float>(width),
                  static_cast<float>(height) };
    m_AnimTime = 1.0f;
}

void CUDARasterizerLoop::OnInit() {
    LoadPipeline();
    //InitCuda();
    LoadAssets();
}

// Load the rendering pipeline dependencies.
void CUDARasterizerLoop::LoadPipeline() {
    UINT dxgiFactoryFlags = 0;
    Device::InitDevice(Win32Application::GetHwnd());

    // Describe and create the command queue.
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    ThrowIfFailed(
        Device::DX12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    // Describe and create the swap chain.
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = FrameCount;
    swapChainDesc.Width = m_width;
    swapChainDesc.Height = m_height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swapChain;
    ThrowIfFailed(Device::DXfactory->CreateSwapChainForHwnd(
        m_commandQueue.Get(),  // Swap chain needs the queue so that it can force
        // a flush on it.
        Win32Application::GetHwnd(), &swapChainDesc, nullptr, nullptr,
        &swapChain));

    // This sample does not support fullscreen transitions.
    //ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(),
    //    DXGI_MWA_NO_ALT_ENTER));

    ThrowIfFailed(swapChain.As(&m_swapChain));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    // Create descriptor heaps.
    {
        // Describe and create a render target view (RTV) descriptor heap.
        D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
        rtvHeapDesc.NumDescriptors = FrameCount;
        rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        ThrowIfFailed(
            Device::DX12Device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

        m_rtvDescriptorSize = Device::DX12Device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc = {};
        samplerHeapDesc.NumDescriptors = 1;
        samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(
            Device::DX12Device->CreateDescriptorHeap(&samplerHeapDesc, IID_PPV_ARGS(&m_samplerHeap)));

        D3D12_DESCRIPTOR_HEAP_DESC textureHeapDesc = {};
        textureHeapDesc.NumDescriptors = 16;
        textureHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        textureHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ThrowIfFailed(
            Device::DX12Device->CreateDescriptorHeap(&textureHeapDesc, IID_PPV_ARGS(&m_textureHeap)));

        D3D12_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.MinLOD = 0;
        samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
        samplerDesc.MipLODBias = 0.0f;
        samplerDesc.MaxAnisotropy = 1;
        samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;

        Device::DX12Device->CreateSampler(&samplerDesc, m_samplerHeap->GetCPUDescriptorHandleForHeapStart());
    }

    // Create frame resources.
    {
        CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
            m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

        // Create a RTV and a command allocator for each frame.
        for (UINT n = 0; n < FrameCount; n++) {
            ThrowIfFailed(
                m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
            Device::DX12Device->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr,
                rtvHandle);
            rtvHandle.Offset(1, m_rtvDescriptorSize);

            ThrowIfFailed(Device::DX12Device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&m_commandAllocators[n])));
        }
    }
}

#define UPPER_ALIGN(A,B) ((UINT)(((A)+((B)-1))&~(B - 1)))
// Load the sample assets.
void CUDARasterizerLoop::LoadAssets() {
    // Create a root signature.
    DXTexturedVertex triangleVertices[] =
    {
        { { 1.0f, 1.0f, 0.0f, 1.0f}, { 1.0f, 0.0f } },
        { { 1.0f, -1.0f, 0.0f, 1.0f}, { 1.0f, 1.0f }},
        { { -1.0f, 1.0f, 0.0f, 1.0f}, { 0.0f, 0.0f } },
        { { -1.0f, -1.0f, 0.0f, 1.0f}, { 0.0f, 1.0f } }
    };

    ColoredVertexData cudaVertices[] = {
        {{1260.0f, 200.0f, 32.0f, 0.0f},{1.0f, 0.0f, 0.0f, 0.0f}},
        {{936.0f, 200.0f, 32.0f, 0.0f},{1.0f, 0.0f, 0.0f, 0.0f}},
        {{936.0f, 150.0f, 0.0f, 0.0f},{1.0f, 0.0f, 0.0f, 0.0f}},
    };
    {
        CD3DX12_DESCRIPTOR_RANGE1 range[3];
        CD3DX12_ROOT_PARAMETER1 parameter[3];
        m_CUDAVertex.allocate_memory(3 * sizeof(ColoredVertexData));
        m_CUDAVertex.copy_from_host(cudaVertices, 3);
        


        range[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
        range[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
        range[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);

        parameter[0].InitAsDescriptorTable(1, range, D3D12_SHADER_VISIBILITY_ALL);

        parameter[1].InitAsDescriptorTable(1, &range[1], D3D12_SHADER_VISIBILITY_ALL);
        parameter[2].InitAsDescriptorTable(1, &range[2], D3D12_SHADER_VISIBILITY_ALL);

        D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
            // Only the input assembler stage needs access to the constant buffer.
            D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
            D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC descRootSignature(_countof(parameter), parameter, 0, nullptr, rootSignatureFlags);
        ComPtr<ID3DBlob> pSignature;
        ComPtr<ID3DBlob> pError;
        ThrowIfFailed(D3D12SerializeVersionedRootSignature(
            &descRootSignature,
            pSignature.GetAddressOf(), pError.GetAddressOf()));

        ThrowIfFailed(Device::DX12Device->CreateRootSignature(
            0, pSignature->GetBufferPointer(), pSignature->GetBufferSize(),
            IID_PPV_ARGS(&m_rootSignature)));
    }
    // Create the pipeline state, which includes compiling and loading shaders.
    {
        ComPtr<ID3DBlob> vertexShader;
        ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
        // Enable better shader debugging with the graphics debugging tools.
        UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
        UINT compileFlags = 0;
#endif
        std::wstring filePath = (L"./MyShaders.hlsl");
        LPCWSTR result = filePath.c_str();
        ThrowIfFailed(D3DCompileFromFile(result, nullptr, nullptr, "VSMain",
            "vs_5_1", compileFlags, 0, &vertexShader,
            nullptr));
        ThrowIfFailed(D3DCompileFromFile(result, nullptr, nullptr, "PSMain",
            "ps_5_1", compileFlags, 0, &pixelShader,
            nullptr));

        D3D12_INPUT_ELEMENT_DESC TexturedVertexInputElementDescs[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
        };

        // Describe and create the graphics pipeline state object (PSO).
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.InputLayout = { TexturedVertexInputElementDescs, _countof(TexturedVertexInputElementDescs) };
        psoDesc.pRootSignature = m_rootSignature.Get();
        psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
        psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.DepthStencilState.DepthEnable = FALSE;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = FrameCount;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;
        ThrowIfFailed(Device::DX12Device->CreateGraphicsPipelineState(
            &psoDesc, IID_PPV_ARGS(&m_pipelineState)));
    }

    // Create the command list.
    ThrowIfFailed(Device::DX12Device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get(),
        IID_PPV_ARGS(&m_commandList)));

    // Command lists are created in the recording state, but there is nothing
    // to record yet. The main loop expects it to be closed, so close it now.
    ThrowIfFailed(m_commandList->Close());

    // Create the vertex buffer.
    {
        const UINT trianglesSize = sizeof(DXTexturedVertex) * 4;



        ThrowIfFailed(Device::DX12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(trianglesSize),
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, nullptr,
            IID_PPV_ARGS(&m_texturedVertex)));
        BYTE* ptr;
        ThrowIfFailed(m_texturedVertex->Map(0, nullptr, (void**)&ptr));
        memcpy(ptr, triangleVertices, sizeof(triangleVertices));
        m_texturedVertex->Unmap(0, nullptr);
        auto inf = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT,
            m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);

        m_texturedBufferView.BufferLocation = m_texturedVertex->GetGPUVirtualAddress();
        m_texturedBufferView.StrideInBytes = sizeof(DXTexturedVertex);
        m_texturedBufferView.SizeInBytes = trianglesSize;
        frameBuffer = new CUDAFrameBuffer(m_width, m_height, 
            m_textureHeap->GetCPUDescriptorHandleForHeapStart(), m_textureHeap->GetGPUDescriptorHandleForHeapStart());
        frameBuffer->Clear(Device::Streams[m_frameIndex], m_frameIndex);
        m_CUDAPipeline.setFrameBufferAndStream(frameBuffer->getWrap(m_frameIndex), Device::Streams[m_frameIndex]);
        m_CUDAPipeline.setPipelineResource(&m_CUDAVertex, nullptr);
        m_CUDAPipeline.setRenderTargetSize(m_width, m_height);
        //SimplePixelShader(m_width, m_height, Device::Streams[m_frameIndex],
        //    m_AnimTime, m_frameIndex, frameBuffer->getRaw(m_frameIndex));
        m_CUDAPipeline.primitiveAssembly(3);
        checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
        m_CUDAPipeline.rasterize(3);
        checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
        frameBuffer->WriteToTex(Device::Streams[m_frameIndex], m_frameIndex);
        checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));   
    }

    // Create synchronization objects and wait until assets have been uploaded to
    // the GPU.
    {
        ThrowIfFailed(Device::DX12Device->CreateFence(m_fenceValues[m_frameIndex],
            D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&m_fence)));

        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;

        memset(&externalSemaphoreHandleDesc, 0,
            sizeof(externalSemaphoreHandleDesc));
        WindowsSecurityAttributes windowsSecurityAttributes;
        LPCWSTR name = NULL;
        HANDLE sharedHandle;
        externalSemaphoreHandleDesc.type =
            cudaExternalSemaphoreHandleTypeD3D12Fence;
        Device::DX12Device->CreateSharedHandle(m_fence.Get(), &windowsSecurityAttributes,
            GENERIC_ALL, name, &sharedHandle);
        externalSemaphoreHandleDesc.handle.win32.handle = sharedHandle;
        externalSemaphoreHandleDesc.flags = 0;

        checkCudaErrors(cudaImportExternalSemaphore(&m_externalSemaphore,
            &externalSemaphoreHandleDesc));

        m_fenceValues[m_frameIndex]++;

        // Create an event handle to use for frame synchronization.
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_fenceEvent == nullptr) {
            ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
        }

        // Wait for the command list to execute; we are reusing the same command
        // list in our main loop but for now, we just want to wait for setup to
        // complete before continuing.
        WaitForGpu();
    }
}

// Render the scene.
void CUDARasterizerLoop::OnRender() {


    // Record all the commands we need to render the scene into the command list.
    PopulateCommandList();

    // Execute the command list.
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present the frame.
    ThrowIfFailed(m_swapChain->Present(1, 0));

    // Schedule a Signal command in the queue.
    const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

    MoveToNextFrame();
}

void CUDARasterizerLoop::OnDestroy() {
    // Ensure that the GPU is no longer referencing resources that are about to be
    // cleaned up by the destructor.
    WaitForGpu();
    if (frameBuffer) delete frameBuffer;
    //ReleaseFrameBuffer();
    m_CUDAVertex.free_memory();
    checkCudaErrors(cudaDestroyExternalSemaphore(m_externalSemaphore));
    //checkCudaErrors(cudaDestroySurfaceObject(m_cuSurface));
    //checkCudaErrors(cudaDestroyExternalMemory(m_externalMemory));
    checkCudaErrors(cudaFree(m_cudaDevVertptr));
    CloseHandle(m_fenceEvent);
}

void CUDARasterizerLoop::PopulateCommandList() {
    ID3D12DescriptorHeap* descriptors[2] = { m_textureHeap.Get(),
    m_samplerHeap.Get() };
    // Command list allocators can only be reset when the associated
    // command lists have finished execution on the GPU; apps should use
    // fences to determine GPU execution progress.
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());

    // However, when ExecuteCommandList() is called on a particular command
    // list, that command list can then be reset at any time and must be before
    // re-recording.
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(),
        m_pipelineState.Get()));

    m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());

    // Set necessary state.
    m_commandList->RSSetViewports(1, &m_viewport);
    m_commandList->RSSetScissorRects(1, &m_scissorRect);

    m_commandList->SetDescriptorHeaps(_countof(descriptors), descriptors);
    m_commandList->SetGraphicsRootDescriptorTable(1, m_textureHeap->GetGPUDescriptorHandleForHeapStart());
    m_commandList->SetGraphicsRootDescriptorTable(2, m_samplerHeap->GetGPUDescriptorHandleForHeapStart());

    // Indicate that the back buffer will be used as a render target.
    m_commandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET));

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
        m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex,
        m_rtvDescriptorSize);
    m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    // Record commands.
    const float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    m_commandList->IASetVertexBuffers(0, 1, &m_texturedBufferView);
    m_commandList->DrawInstanced(4, 1, 0, 0);

    // Indicate that the back buffer will now be used to present.
    m_commandList->ResourceBarrier(
        1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_renderTargets[m_frameIndex].Get(),
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    ThrowIfFailed(m_commandList->Close());
}

// Wait for pending GPU work to complete.
void CUDARasterizerLoop::WaitForGpu() {
    // Schedule a Signal command in the queue.
    ThrowIfFailed(
        m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));

    // Wait until the fence has been processed.
    ThrowIfFailed(
        m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
    WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);

    // Increment the fence value for the current frame.
    m_fenceValues[m_frameIndex]++;
}

// Prepare to render the next frame.
void CUDARasterizerLoop::MoveToNextFrame() {
    const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;
    memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

    externalSemaphoreWaitParams.params.fence.value = currentFenceValue;
    externalSemaphoreWaitParams.flags = 0;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(
        &m_externalSemaphore, &externalSemaphoreWaitParams, 1, Device::Streams[m_frameIndex]));

    m_AnimTime += 0.01f;
    //CUDAWriteToTex(m_width, m_height, m_cuSurface,
    //    Device::Streams[m_frameIndex], m_AnimTime, m_frameIndex);
    frameBuffer->Clear(Device::Streams[m_frameIndex], m_frameIndex);
    checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
    m_CUDAPipeline.setRenderTargetSize(m_width, m_height);
    m_CUDAPipeline.setFrameBufferAndStream(frameBuffer->getWrap(m_frameIndex), Device::Streams[m_frameIndex]);
    SimplePixelShader(m_width, m_height, Device::Streams[m_frameIndex],
        m_AnimTime, m_frameIndex, frameBuffer->getRaw(m_frameIndex));
    checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
    m_CUDAPipeline.primitiveAssembly(3);
    checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
    m_CUDAPipeline.rasterize(3);
    checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
    frameBuffer->WriteToTex(Device::Streams[m_frameIndex], m_frameIndex);
    //checkCudaErrors(cudaStreamSynchronize(Device::Streams[m_frameIndex]));
    cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
    memset(&externalSemaphoreSignalParams, 0,
        sizeof(externalSemaphoreSignalParams));
    m_fenceValues[m_frameIndex] = currentFenceValue + 1;
    externalSemaphoreSignalParams.params.fence.value =
        m_fenceValues[m_frameIndex];
    externalSemaphoreSignalParams.flags = 0;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(
        &m_externalSemaphore, &externalSemaphoreSignalParams, 1, Device::Streams[(m_frameIndex + 1) % 3]));

    // Update the frame index.
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    // If the next frame is not ready to be rendered yet, wait until it is ready.
    if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex]) {
        ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex],
            m_fenceEvent));
        WaitForSingleObjectEx(m_fenceEvent, INFINITE, FALSE);
    }

    // Set the fence value for the next frame.
    m_fenceValues[m_frameIndex] = currentFenceValue + 2;
}
