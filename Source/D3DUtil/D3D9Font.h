/*
 * (C) 2019 see Authors.txt
 *
 * This file is part of MPC-BE.
 *
 * MPC-BE is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * MPC-BE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include "D3DCommon.h"

// Font creation flags
#define D3DFONT_ZENABLE     0x0004

// Font rendering flags
#define D3DFONT_CENTERED_X  0x0001
#define D3DFONT_CENTERED_Y  0x0002
#define D3DFONT_FILTERED    0x0008

class CD3D9Font
{
	// Font properties
	WCHAR m_strFontName[80];
	DWORD m_dwFontHeight;
	DWORD m_dwFontFlags;

	WCHAR m_Characters[128];
	FloatRect m_fTexCoords[128] = {};

	IDirect3DDevice9*       m_pd3dDevice    = nullptr; // A D3DDevice used for rendering
	IDirect3DTexture9*      m_pTexture      = nullptr; // The d3d texture for this font
	IDirect3DVertexBuffer9* m_pVertexBuffer = nullptr; // VertexBuffer for rendering text
	UINT  m_uTexWidth  = 0;                   // Texture dimensions
	UINT  m_uTexHeight = 0;
	float m_fTextScale = 1.0f;

	// Stateblocks for setting and restoring render states
	IDirect3DStateBlock9* m_pStateBlockSaved    = nullptr;
	IDirect3DStateBlock9* m_pStateBlockDrawText = nullptr;
	HRESULT CreateStateBlocks();

public:
	// Constructor / destructor
	CD3D9Font(const WCHAR* strFontName, const DWORD dwHeight, const DWORD dwFlags=0L);
	~CD3D9Font();

	// Initializing and destroying device-dependent objects
	HRESULT InitDeviceObjects(IDirect3DDevice9* pd3dDevice);
	void InvalidateDeviceObjects();

	// Function to get extent of text
	HRESULT GetTextExtent(const WCHAR* strText, SIZE* pSize);

	// 2D text drawing function
	HRESULT Draw2DText(float sx, float sy, const D3DCOLOR color, const WCHAR* strText, const DWORD dwFlags=0L);
};
