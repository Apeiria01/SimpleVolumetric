

#include "stdafx.h"
#include "DX12WindowLoop.h"

using namespace Microsoft::WRL;

DX12WindowLoop::DX12WindowLoop(UINT width, UINT height, std::string name)
    : m_width(width), m_height(height), m_title(name), m_useWarpDevice(false) {
  m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);
}

DX12WindowLoop::~DX12WindowLoop() {}

std::wstring DX12WindowLoop::string2wstring(const std::string& s) {
  int len;
  int slength = (int)s.length() + 1;
  len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
  wchar_t* buf = new wchar_t[len];
  MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
  std::wstring r(buf);
  delete[] buf;
  return r;
}

_Use_decl_annotations_ void DX12WindowLoop::GetHardwareAdapter(
    IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter) {
  ComPtr<IDXGIAdapter1> adapter;
  *ppAdapter = nullptr;

  for (UINT adapterIndex = 0;
       DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter);
       ++adapterIndex) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      // Don't select the Basic Render Driver adapter.
      // If you want a software adapter, pass in "/warp" on the command line.
      continue;
    }

    // Check to see if the adapter supports Direct3D 12, but don't create the
    // actual device yet.
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_1,
                                    _uuidof(ID3D12Device), nullptr))) {
      break;
    }
  }

  *ppAdapter = adapter.Detach();
}

// Helper function for setting the window's title text.
void DX12WindowLoop::SetCustomWindowText(const char* text) {
  std::string windowText = m_title + text;
  SetWindowText(Win32Application::GetHwnd(), windowText.c_str());
}

// Helper function for parsing any supplied command line args.
_Use_decl_annotations_ void DX12WindowLoop::ParseCommandLineArgs(WCHAR* argv[],
                                                                 int argc) {
  for (int i = 1; i < argc; ++i) {
    if (_wcsnicmp(argv[i], L"-warp", wcslen(argv[i])) == 0 ||
        _wcsnicmp(argv[i], L"/warp", wcslen(argv[i])) == 0) {
      m_useWarpDevice = true;
      m_title = m_title + " (WARP)";
    }
  }
}
