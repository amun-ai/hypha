(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define("hyphaWebsocketClient", [], factory);
	else if(typeof exports === 'object')
		exports["hyphaWebsocketClient"] = factory();
	else
		root["hyphaWebsocketClient"] = factory();
})(this, () => {
return /******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "./node_modules/tweetnacl/nacl-fast.js":
/*!*********************************************!*\
  !*** ./node_modules/tweetnacl/nacl-fast.js ***!
  \*********************************************/
/***/ ((module, __unused_webpack_exports, __webpack_require__) => {

(function(nacl) {
'use strict';

// Ported in 2014 by Dmitry Chestnykh and Devi Mandiri.
// Public domain.
//
// Implementation derived from TweetNaCl version 20140427.
// See for details: http://tweetnacl.cr.yp.to/

var gf = function(init) {
  var i, r = new Float64Array(16);
  if (init) for (i = 0; i < init.length; i++) r[i] = init[i];
  return r;
};

//  Pluggable, initialized in high-level API below.
var randombytes = function(/* x, n */) { throw new Error('no PRNG'); };

var _0 = new Uint8Array(16);
var _9 = new Uint8Array(32); _9[0] = 9;

var gf0 = gf(),
    gf1 = gf([1]),
    _121665 = gf([0xdb41, 1]),
    D = gf([0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070, 0xe898, 0x7779, 0x4079, 0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203]),
    D2 = gf([0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0, 0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406]),
    X = gf([0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c, 0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169]),
    Y = gf([0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666]),
    I = gf([0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43, 0xd7a7, 0x3dfb, 0x0099, 0x2b4d, 0xdf0b, 0x4fc1, 0x2480, 0x2b83]);

function ts64(x, i, h, l) {
  x[i]   = (h >> 24) & 0xff;
  x[i+1] = (h >> 16) & 0xff;
  x[i+2] = (h >>  8) & 0xff;
  x[i+3] = h & 0xff;
  x[i+4] = (l >> 24)  & 0xff;
  x[i+5] = (l >> 16)  & 0xff;
  x[i+6] = (l >>  8)  & 0xff;
  x[i+7] = l & 0xff;
}

function vn(x, xi, y, yi, n) {
  var i,d = 0;
  for (i = 0; i < n; i++) d |= x[xi+i]^y[yi+i];
  return (1 & ((d - 1) >>> 8)) - 1;
}

function crypto_verify_16(x, xi, y, yi) {
  return vn(x,xi,y,yi,16);
}

function crypto_verify_32(x, xi, y, yi) {
  return vn(x,xi,y,yi,32);
}

function core_salsa20(o, p, k, c) {
  var j0  = c[ 0] & 0xff | (c[ 1] & 0xff)<<8 | (c[ 2] & 0xff)<<16 | (c[ 3] & 0xff)<<24,
      j1  = k[ 0] & 0xff | (k[ 1] & 0xff)<<8 | (k[ 2] & 0xff)<<16 | (k[ 3] & 0xff)<<24,
      j2  = k[ 4] & 0xff | (k[ 5] & 0xff)<<8 | (k[ 6] & 0xff)<<16 | (k[ 7] & 0xff)<<24,
      j3  = k[ 8] & 0xff | (k[ 9] & 0xff)<<8 | (k[10] & 0xff)<<16 | (k[11] & 0xff)<<24,
      j4  = k[12] & 0xff | (k[13] & 0xff)<<8 | (k[14] & 0xff)<<16 | (k[15] & 0xff)<<24,
      j5  = c[ 4] & 0xff | (c[ 5] & 0xff)<<8 | (c[ 6] & 0xff)<<16 | (c[ 7] & 0xff)<<24,
      j6  = p[ 0] & 0xff | (p[ 1] & 0xff)<<8 | (p[ 2] & 0xff)<<16 | (p[ 3] & 0xff)<<24,
      j7  = p[ 4] & 0xff | (p[ 5] & 0xff)<<8 | (p[ 6] & 0xff)<<16 | (p[ 7] & 0xff)<<24,
      j8  = p[ 8] & 0xff | (p[ 9] & 0xff)<<8 | (p[10] & 0xff)<<16 | (p[11] & 0xff)<<24,
      j9  = p[12] & 0xff | (p[13] & 0xff)<<8 | (p[14] & 0xff)<<16 | (p[15] & 0xff)<<24,
      j10 = c[ 8] & 0xff | (c[ 9] & 0xff)<<8 | (c[10] & 0xff)<<16 | (c[11] & 0xff)<<24,
      j11 = k[16] & 0xff | (k[17] & 0xff)<<8 | (k[18] & 0xff)<<16 | (k[19] & 0xff)<<24,
      j12 = k[20] & 0xff | (k[21] & 0xff)<<8 | (k[22] & 0xff)<<16 | (k[23] & 0xff)<<24,
      j13 = k[24] & 0xff | (k[25] & 0xff)<<8 | (k[26] & 0xff)<<16 | (k[27] & 0xff)<<24,
      j14 = k[28] & 0xff | (k[29] & 0xff)<<8 | (k[30] & 0xff)<<16 | (k[31] & 0xff)<<24,
      j15 = c[12] & 0xff | (c[13] & 0xff)<<8 | (c[14] & 0xff)<<16 | (c[15] & 0xff)<<24;

  var x0 = j0, x1 = j1, x2 = j2, x3 = j3, x4 = j4, x5 = j5, x6 = j6, x7 = j7,
      x8 = j8, x9 = j9, x10 = j10, x11 = j11, x12 = j12, x13 = j13, x14 = j14,
      x15 = j15, u;

  for (var i = 0; i < 20; i += 2) {
    u = x0 + x12 | 0;
    x4 ^= u<<7 | u>>>(32-7);
    u = x4 + x0 | 0;
    x8 ^= u<<9 | u>>>(32-9);
    u = x8 + x4 | 0;
    x12 ^= u<<13 | u>>>(32-13);
    u = x12 + x8 | 0;
    x0 ^= u<<18 | u>>>(32-18);

    u = x5 + x1 | 0;
    x9 ^= u<<7 | u>>>(32-7);
    u = x9 + x5 | 0;
    x13 ^= u<<9 | u>>>(32-9);
    u = x13 + x9 | 0;
    x1 ^= u<<13 | u>>>(32-13);
    u = x1 + x13 | 0;
    x5 ^= u<<18 | u>>>(32-18);

    u = x10 + x6 | 0;
    x14 ^= u<<7 | u>>>(32-7);
    u = x14 + x10 | 0;
    x2 ^= u<<9 | u>>>(32-9);
    u = x2 + x14 | 0;
    x6 ^= u<<13 | u>>>(32-13);
    u = x6 + x2 | 0;
    x10 ^= u<<18 | u>>>(32-18);

    u = x15 + x11 | 0;
    x3 ^= u<<7 | u>>>(32-7);
    u = x3 + x15 | 0;
    x7 ^= u<<9 | u>>>(32-9);
    u = x7 + x3 | 0;
    x11 ^= u<<13 | u>>>(32-13);
    u = x11 + x7 | 0;
    x15 ^= u<<18 | u>>>(32-18);

    u = x0 + x3 | 0;
    x1 ^= u<<7 | u>>>(32-7);
    u = x1 + x0 | 0;
    x2 ^= u<<9 | u>>>(32-9);
    u = x2 + x1 | 0;
    x3 ^= u<<13 | u>>>(32-13);
    u = x3 + x2 | 0;
    x0 ^= u<<18 | u>>>(32-18);

    u = x5 + x4 | 0;
    x6 ^= u<<7 | u>>>(32-7);
    u = x6 + x5 | 0;
    x7 ^= u<<9 | u>>>(32-9);
    u = x7 + x6 | 0;
    x4 ^= u<<13 | u>>>(32-13);
    u = x4 + x7 | 0;
    x5 ^= u<<18 | u>>>(32-18);

    u = x10 + x9 | 0;
    x11 ^= u<<7 | u>>>(32-7);
    u = x11 + x10 | 0;
    x8 ^= u<<9 | u>>>(32-9);
    u = x8 + x11 | 0;
    x9 ^= u<<13 | u>>>(32-13);
    u = x9 + x8 | 0;
    x10 ^= u<<18 | u>>>(32-18);

    u = x15 + x14 | 0;
    x12 ^= u<<7 | u>>>(32-7);
    u = x12 + x15 | 0;
    x13 ^= u<<9 | u>>>(32-9);
    u = x13 + x12 | 0;
    x14 ^= u<<13 | u>>>(32-13);
    u = x14 + x13 | 0;
    x15 ^= u<<18 | u>>>(32-18);
  }
   x0 =  x0 +  j0 | 0;
   x1 =  x1 +  j1 | 0;
   x2 =  x2 +  j2 | 0;
   x3 =  x3 +  j3 | 0;
   x4 =  x4 +  j4 | 0;
   x5 =  x5 +  j5 | 0;
   x6 =  x6 +  j6 | 0;
   x7 =  x7 +  j7 | 0;
   x8 =  x8 +  j8 | 0;
   x9 =  x9 +  j9 | 0;
  x10 = x10 + j10 | 0;
  x11 = x11 + j11 | 0;
  x12 = x12 + j12 | 0;
  x13 = x13 + j13 | 0;
  x14 = x14 + j14 | 0;
  x15 = x15 + j15 | 0;

  o[ 0] = x0 >>>  0 & 0xff;
  o[ 1] = x0 >>>  8 & 0xff;
  o[ 2] = x0 >>> 16 & 0xff;
  o[ 3] = x0 >>> 24 & 0xff;

  o[ 4] = x1 >>>  0 & 0xff;
  o[ 5] = x1 >>>  8 & 0xff;
  o[ 6] = x1 >>> 16 & 0xff;
  o[ 7] = x1 >>> 24 & 0xff;

  o[ 8] = x2 >>>  0 & 0xff;
  o[ 9] = x2 >>>  8 & 0xff;
  o[10] = x2 >>> 16 & 0xff;
  o[11] = x2 >>> 24 & 0xff;

  o[12] = x3 >>>  0 & 0xff;
  o[13] = x3 >>>  8 & 0xff;
  o[14] = x3 >>> 16 & 0xff;
  o[15] = x3 >>> 24 & 0xff;

  o[16] = x4 >>>  0 & 0xff;
  o[17] = x4 >>>  8 & 0xff;
  o[18] = x4 >>> 16 & 0xff;
  o[19] = x4 >>> 24 & 0xff;

  o[20] = x5 >>>  0 & 0xff;
  o[21] = x5 >>>  8 & 0xff;
  o[22] = x5 >>> 16 & 0xff;
  o[23] = x5 >>> 24 & 0xff;

  o[24] = x6 >>>  0 & 0xff;
  o[25] = x6 >>>  8 & 0xff;
  o[26] = x6 >>> 16 & 0xff;
  o[27] = x6 >>> 24 & 0xff;

  o[28] = x7 >>>  0 & 0xff;
  o[29] = x7 >>>  8 & 0xff;
  o[30] = x7 >>> 16 & 0xff;
  o[31] = x7 >>> 24 & 0xff;

  o[32] = x8 >>>  0 & 0xff;
  o[33] = x8 >>>  8 & 0xff;
  o[34] = x8 >>> 16 & 0xff;
  o[35] = x8 >>> 24 & 0xff;

  o[36] = x9 >>>  0 & 0xff;
  o[37] = x9 >>>  8 & 0xff;
  o[38] = x9 >>> 16 & 0xff;
  o[39] = x9 >>> 24 & 0xff;

  o[40] = x10 >>>  0 & 0xff;
  o[41] = x10 >>>  8 & 0xff;
  o[42] = x10 >>> 16 & 0xff;
  o[43] = x10 >>> 24 & 0xff;

  o[44] = x11 >>>  0 & 0xff;
  o[45] = x11 >>>  8 & 0xff;
  o[46] = x11 >>> 16 & 0xff;
  o[47] = x11 >>> 24 & 0xff;

  o[48] = x12 >>>  0 & 0xff;
  o[49] = x12 >>>  8 & 0xff;
  o[50] = x12 >>> 16 & 0xff;
  o[51] = x12 >>> 24 & 0xff;

  o[52] = x13 >>>  0 & 0xff;
  o[53] = x13 >>>  8 & 0xff;
  o[54] = x13 >>> 16 & 0xff;
  o[55] = x13 >>> 24 & 0xff;

  o[56] = x14 >>>  0 & 0xff;
  o[57] = x14 >>>  8 & 0xff;
  o[58] = x14 >>> 16 & 0xff;
  o[59] = x14 >>> 24 & 0xff;

  o[60] = x15 >>>  0 & 0xff;
  o[61] = x15 >>>  8 & 0xff;
  o[62] = x15 >>> 16 & 0xff;
  o[63] = x15 >>> 24 & 0xff;
}

function core_hsalsa20(o,p,k,c) {
  var j0  = c[ 0] & 0xff | (c[ 1] & 0xff)<<8 | (c[ 2] & 0xff)<<16 | (c[ 3] & 0xff)<<24,
      j1  = k[ 0] & 0xff | (k[ 1] & 0xff)<<8 | (k[ 2] & 0xff)<<16 | (k[ 3] & 0xff)<<24,
      j2  = k[ 4] & 0xff | (k[ 5] & 0xff)<<8 | (k[ 6] & 0xff)<<16 | (k[ 7] & 0xff)<<24,
      j3  = k[ 8] & 0xff | (k[ 9] & 0xff)<<8 | (k[10] & 0xff)<<16 | (k[11] & 0xff)<<24,
      j4  = k[12] & 0xff | (k[13] & 0xff)<<8 | (k[14] & 0xff)<<16 | (k[15] & 0xff)<<24,
      j5  = c[ 4] & 0xff | (c[ 5] & 0xff)<<8 | (c[ 6] & 0xff)<<16 | (c[ 7] & 0xff)<<24,
      j6  = p[ 0] & 0xff | (p[ 1] & 0xff)<<8 | (p[ 2] & 0xff)<<16 | (p[ 3] & 0xff)<<24,
      j7  = p[ 4] & 0xff | (p[ 5] & 0xff)<<8 | (p[ 6] & 0xff)<<16 | (p[ 7] & 0xff)<<24,
      j8  = p[ 8] & 0xff | (p[ 9] & 0xff)<<8 | (p[10] & 0xff)<<16 | (p[11] & 0xff)<<24,
      j9  = p[12] & 0xff | (p[13] & 0xff)<<8 | (p[14] & 0xff)<<16 | (p[15] & 0xff)<<24,
      j10 = c[ 8] & 0xff | (c[ 9] & 0xff)<<8 | (c[10] & 0xff)<<16 | (c[11] & 0xff)<<24,
      j11 = k[16] & 0xff | (k[17] & 0xff)<<8 | (k[18] & 0xff)<<16 | (k[19] & 0xff)<<24,
      j12 = k[20] & 0xff | (k[21] & 0xff)<<8 | (k[22] & 0xff)<<16 | (k[23] & 0xff)<<24,
      j13 = k[24] & 0xff | (k[25] & 0xff)<<8 | (k[26] & 0xff)<<16 | (k[27] & 0xff)<<24,
      j14 = k[28] & 0xff | (k[29] & 0xff)<<8 | (k[30] & 0xff)<<16 | (k[31] & 0xff)<<24,
      j15 = c[12] & 0xff | (c[13] & 0xff)<<8 | (c[14] & 0xff)<<16 | (c[15] & 0xff)<<24;

  var x0 = j0, x1 = j1, x2 = j2, x3 = j3, x4 = j4, x5 = j5, x6 = j6, x7 = j7,
      x8 = j8, x9 = j9, x10 = j10, x11 = j11, x12 = j12, x13 = j13, x14 = j14,
      x15 = j15, u;

  for (var i = 0; i < 20; i += 2) {
    u = x0 + x12 | 0;
    x4 ^= u<<7 | u>>>(32-7);
    u = x4 + x0 | 0;
    x8 ^= u<<9 | u>>>(32-9);
    u = x8 + x4 | 0;
    x12 ^= u<<13 | u>>>(32-13);
    u = x12 + x8 | 0;
    x0 ^= u<<18 | u>>>(32-18);

    u = x5 + x1 | 0;
    x9 ^= u<<7 | u>>>(32-7);
    u = x9 + x5 | 0;
    x13 ^= u<<9 | u>>>(32-9);
    u = x13 + x9 | 0;
    x1 ^= u<<13 | u>>>(32-13);
    u = x1 + x13 | 0;
    x5 ^= u<<18 | u>>>(32-18);

    u = x10 + x6 | 0;
    x14 ^= u<<7 | u>>>(32-7);
    u = x14 + x10 | 0;
    x2 ^= u<<9 | u>>>(32-9);
    u = x2 + x14 | 0;
    x6 ^= u<<13 | u>>>(32-13);
    u = x6 + x2 | 0;
    x10 ^= u<<18 | u>>>(32-18);

    u = x15 + x11 | 0;
    x3 ^= u<<7 | u>>>(32-7);
    u = x3 + x15 | 0;
    x7 ^= u<<9 | u>>>(32-9);
    u = x7 + x3 | 0;
    x11 ^= u<<13 | u>>>(32-13);
    u = x11 + x7 | 0;
    x15 ^= u<<18 | u>>>(32-18);

    u = x0 + x3 | 0;
    x1 ^= u<<7 | u>>>(32-7);
    u = x1 + x0 | 0;
    x2 ^= u<<9 | u>>>(32-9);
    u = x2 + x1 | 0;
    x3 ^= u<<13 | u>>>(32-13);
    u = x3 + x2 | 0;
    x0 ^= u<<18 | u>>>(32-18);

    u = x5 + x4 | 0;
    x6 ^= u<<7 | u>>>(32-7);
    u = x6 + x5 | 0;
    x7 ^= u<<9 | u>>>(32-9);
    u = x7 + x6 | 0;
    x4 ^= u<<13 | u>>>(32-13);
    u = x4 + x7 | 0;
    x5 ^= u<<18 | u>>>(32-18);

    u = x10 + x9 | 0;
    x11 ^= u<<7 | u>>>(32-7);
    u = x11 + x10 | 0;
    x8 ^= u<<9 | u>>>(32-9);
    u = x8 + x11 | 0;
    x9 ^= u<<13 | u>>>(32-13);
    u = x9 + x8 | 0;
    x10 ^= u<<18 | u>>>(32-18);

    u = x15 + x14 | 0;
    x12 ^= u<<7 | u>>>(32-7);
    u = x12 + x15 | 0;
    x13 ^= u<<9 | u>>>(32-9);
    u = x13 + x12 | 0;
    x14 ^= u<<13 | u>>>(32-13);
    u = x14 + x13 | 0;
    x15 ^= u<<18 | u>>>(32-18);
  }

  o[ 0] = x0 >>>  0 & 0xff;
  o[ 1] = x0 >>>  8 & 0xff;
  o[ 2] = x0 >>> 16 & 0xff;
  o[ 3] = x0 >>> 24 & 0xff;

  o[ 4] = x5 >>>  0 & 0xff;
  o[ 5] = x5 >>>  8 & 0xff;
  o[ 6] = x5 >>> 16 & 0xff;
  o[ 7] = x5 >>> 24 & 0xff;

  o[ 8] = x10 >>>  0 & 0xff;
  o[ 9] = x10 >>>  8 & 0xff;
  o[10] = x10 >>> 16 & 0xff;
  o[11] = x10 >>> 24 & 0xff;

  o[12] = x15 >>>  0 & 0xff;
  o[13] = x15 >>>  8 & 0xff;
  o[14] = x15 >>> 16 & 0xff;
  o[15] = x15 >>> 24 & 0xff;

  o[16] = x6 >>>  0 & 0xff;
  o[17] = x6 >>>  8 & 0xff;
  o[18] = x6 >>> 16 & 0xff;
  o[19] = x6 >>> 24 & 0xff;

  o[20] = x7 >>>  0 & 0xff;
  o[21] = x7 >>>  8 & 0xff;
  o[22] = x7 >>> 16 & 0xff;
  o[23] = x7 >>> 24 & 0xff;

  o[24] = x8 >>>  0 & 0xff;
  o[25] = x8 >>>  8 & 0xff;
  o[26] = x8 >>> 16 & 0xff;
  o[27] = x8 >>> 24 & 0xff;

  o[28] = x9 >>>  0 & 0xff;
  o[29] = x9 >>>  8 & 0xff;
  o[30] = x9 >>> 16 & 0xff;
  o[31] = x9 >>> 24 & 0xff;
}

function crypto_core_salsa20(out,inp,k,c) {
  core_salsa20(out,inp,k,c);
}

function crypto_core_hsalsa20(out,inp,k,c) {
  core_hsalsa20(out,inp,k,c);
}

var sigma = new Uint8Array([101, 120, 112, 97, 110, 100, 32, 51, 50, 45, 98, 121, 116, 101, 32, 107]);
            // "expand 32-byte k"

function crypto_stream_salsa20_xor(c,cpos,m,mpos,b,n,k) {
  var z = new Uint8Array(16), x = new Uint8Array(64);
  var u, i;
  for (i = 0; i < 16; i++) z[i] = 0;
  for (i = 0; i < 8; i++) z[i] = n[i];
  while (b >= 64) {
    crypto_core_salsa20(x,z,k,sigma);
    for (i = 0; i < 64; i++) c[cpos+i] = m[mpos+i] ^ x[i];
    u = 1;
    for (i = 8; i < 16; i++) {
      u = u + (z[i] & 0xff) | 0;
      z[i] = u & 0xff;
      u >>>= 8;
    }
    b -= 64;
    cpos += 64;
    mpos += 64;
  }
  if (b > 0) {
    crypto_core_salsa20(x,z,k,sigma);
    for (i = 0; i < b; i++) c[cpos+i] = m[mpos+i] ^ x[i];
  }
  return 0;
}

function crypto_stream_salsa20(c,cpos,b,n,k) {
  var z = new Uint8Array(16), x = new Uint8Array(64);
  var u, i;
  for (i = 0; i < 16; i++) z[i] = 0;
  for (i = 0; i < 8; i++) z[i] = n[i];
  while (b >= 64) {
    crypto_core_salsa20(x,z,k,sigma);
    for (i = 0; i < 64; i++) c[cpos+i] = x[i];
    u = 1;
    for (i = 8; i < 16; i++) {
      u = u + (z[i] & 0xff) | 0;
      z[i] = u & 0xff;
      u >>>= 8;
    }
    b -= 64;
    cpos += 64;
  }
  if (b > 0) {
    crypto_core_salsa20(x,z,k,sigma);
    for (i = 0; i < b; i++) c[cpos+i] = x[i];
  }
  return 0;
}

function crypto_stream(c,cpos,d,n,k) {
  var s = new Uint8Array(32);
  crypto_core_hsalsa20(s,n,k,sigma);
  var sn = new Uint8Array(8);
  for (var i = 0; i < 8; i++) sn[i] = n[i+16];
  return crypto_stream_salsa20(c,cpos,d,sn,s);
}

function crypto_stream_xor(c,cpos,m,mpos,d,n,k) {
  var s = new Uint8Array(32);
  crypto_core_hsalsa20(s,n,k,sigma);
  var sn = new Uint8Array(8);
  for (var i = 0; i < 8; i++) sn[i] = n[i+16];
  return crypto_stream_salsa20_xor(c,cpos,m,mpos,d,sn,s);
}

/*
* Port of Andrew Moon's Poly1305-donna-16. Public domain.
* https://github.com/floodyberry/poly1305-donna
*/

var poly1305 = function(key) {
  this.buffer = new Uint8Array(16);
  this.r = new Uint16Array(10);
  this.h = new Uint16Array(10);
  this.pad = new Uint16Array(8);
  this.leftover = 0;
  this.fin = 0;

  var t0, t1, t2, t3, t4, t5, t6, t7;

  t0 = key[ 0] & 0xff | (key[ 1] & 0xff) << 8; this.r[0] = ( t0                     ) & 0x1fff;
  t1 = key[ 2] & 0xff | (key[ 3] & 0xff) << 8; this.r[1] = ((t0 >>> 13) | (t1 <<  3)) & 0x1fff;
  t2 = key[ 4] & 0xff | (key[ 5] & 0xff) << 8; this.r[2] = ((t1 >>> 10) | (t2 <<  6)) & 0x1f03;
  t3 = key[ 6] & 0xff | (key[ 7] & 0xff) << 8; this.r[3] = ((t2 >>>  7) | (t3 <<  9)) & 0x1fff;
  t4 = key[ 8] & 0xff | (key[ 9] & 0xff) << 8; this.r[4] = ((t3 >>>  4) | (t4 << 12)) & 0x00ff;
  this.r[5] = ((t4 >>>  1)) & 0x1ffe;
  t5 = key[10] & 0xff | (key[11] & 0xff) << 8; this.r[6] = ((t4 >>> 14) | (t5 <<  2)) & 0x1fff;
  t6 = key[12] & 0xff | (key[13] & 0xff) << 8; this.r[7] = ((t5 >>> 11) | (t6 <<  5)) & 0x1f81;
  t7 = key[14] & 0xff | (key[15] & 0xff) << 8; this.r[8] = ((t6 >>>  8) | (t7 <<  8)) & 0x1fff;
  this.r[9] = ((t7 >>>  5)) & 0x007f;

  this.pad[0] = key[16] & 0xff | (key[17] & 0xff) << 8;
  this.pad[1] = key[18] & 0xff | (key[19] & 0xff) << 8;
  this.pad[2] = key[20] & 0xff | (key[21] & 0xff) << 8;
  this.pad[3] = key[22] & 0xff | (key[23] & 0xff) << 8;
  this.pad[4] = key[24] & 0xff | (key[25] & 0xff) << 8;
  this.pad[5] = key[26] & 0xff | (key[27] & 0xff) << 8;
  this.pad[6] = key[28] & 0xff | (key[29] & 0xff) << 8;
  this.pad[7] = key[30] & 0xff | (key[31] & 0xff) << 8;
};

poly1305.prototype.blocks = function(m, mpos, bytes) {
  var hibit = this.fin ? 0 : (1 << 11);
  var t0, t1, t2, t3, t4, t5, t6, t7, c;
  var d0, d1, d2, d3, d4, d5, d6, d7, d8, d9;

  var h0 = this.h[0],
      h1 = this.h[1],
      h2 = this.h[2],
      h3 = this.h[3],
      h4 = this.h[4],
      h5 = this.h[5],
      h6 = this.h[6],
      h7 = this.h[7],
      h8 = this.h[8],
      h9 = this.h[9];

  var r0 = this.r[0],
      r1 = this.r[1],
      r2 = this.r[2],
      r3 = this.r[3],
      r4 = this.r[4],
      r5 = this.r[5],
      r6 = this.r[6],
      r7 = this.r[7],
      r8 = this.r[8],
      r9 = this.r[9];

  while (bytes >= 16) {
    t0 = m[mpos+ 0] & 0xff | (m[mpos+ 1] & 0xff) << 8; h0 += ( t0                     ) & 0x1fff;
    t1 = m[mpos+ 2] & 0xff | (m[mpos+ 3] & 0xff) << 8; h1 += ((t0 >>> 13) | (t1 <<  3)) & 0x1fff;
    t2 = m[mpos+ 4] & 0xff | (m[mpos+ 5] & 0xff) << 8; h2 += ((t1 >>> 10) | (t2 <<  6)) & 0x1fff;
    t3 = m[mpos+ 6] & 0xff | (m[mpos+ 7] & 0xff) << 8; h3 += ((t2 >>>  7) | (t3 <<  9)) & 0x1fff;
    t4 = m[mpos+ 8] & 0xff | (m[mpos+ 9] & 0xff) << 8; h4 += ((t3 >>>  4) | (t4 << 12)) & 0x1fff;
    h5 += ((t4 >>>  1)) & 0x1fff;
    t5 = m[mpos+10] & 0xff | (m[mpos+11] & 0xff) << 8; h6 += ((t4 >>> 14) | (t5 <<  2)) & 0x1fff;
    t6 = m[mpos+12] & 0xff | (m[mpos+13] & 0xff) << 8; h7 += ((t5 >>> 11) | (t6 <<  5)) & 0x1fff;
    t7 = m[mpos+14] & 0xff | (m[mpos+15] & 0xff) << 8; h8 += ((t6 >>>  8) | (t7 <<  8)) & 0x1fff;
    h9 += ((t7 >>> 5)) | hibit;

    c = 0;

    d0 = c;
    d0 += h0 * r0;
    d0 += h1 * (5 * r9);
    d0 += h2 * (5 * r8);
    d0 += h3 * (5 * r7);
    d0 += h4 * (5 * r6);
    c = (d0 >>> 13); d0 &= 0x1fff;
    d0 += h5 * (5 * r5);
    d0 += h6 * (5 * r4);
    d0 += h7 * (5 * r3);
    d0 += h8 * (5 * r2);
    d0 += h9 * (5 * r1);
    c += (d0 >>> 13); d0 &= 0x1fff;

    d1 = c;
    d1 += h0 * r1;
    d1 += h1 * r0;
    d1 += h2 * (5 * r9);
    d1 += h3 * (5 * r8);
    d1 += h4 * (5 * r7);
    c = (d1 >>> 13); d1 &= 0x1fff;
    d1 += h5 * (5 * r6);
    d1 += h6 * (5 * r5);
    d1 += h7 * (5 * r4);
    d1 += h8 * (5 * r3);
    d1 += h9 * (5 * r2);
    c += (d1 >>> 13); d1 &= 0x1fff;

    d2 = c;
    d2 += h0 * r2;
    d2 += h1 * r1;
    d2 += h2 * r0;
    d2 += h3 * (5 * r9);
    d2 += h4 * (5 * r8);
    c = (d2 >>> 13); d2 &= 0x1fff;
    d2 += h5 * (5 * r7);
    d2 += h6 * (5 * r6);
    d2 += h7 * (5 * r5);
    d2 += h8 * (5 * r4);
    d2 += h9 * (5 * r3);
    c += (d2 >>> 13); d2 &= 0x1fff;

    d3 = c;
    d3 += h0 * r3;
    d3 += h1 * r2;
    d3 += h2 * r1;
    d3 += h3 * r0;
    d3 += h4 * (5 * r9);
    c = (d3 >>> 13); d3 &= 0x1fff;
    d3 += h5 * (5 * r8);
    d3 += h6 * (5 * r7);
    d3 += h7 * (5 * r6);
    d3 += h8 * (5 * r5);
    d3 += h9 * (5 * r4);
    c += (d3 >>> 13); d3 &= 0x1fff;

    d4 = c;
    d4 += h0 * r4;
    d4 += h1 * r3;
    d4 += h2 * r2;
    d4 += h3 * r1;
    d4 += h4 * r0;
    c = (d4 >>> 13); d4 &= 0x1fff;
    d4 += h5 * (5 * r9);
    d4 += h6 * (5 * r8);
    d4 += h7 * (5 * r7);
    d4 += h8 * (5 * r6);
    d4 += h9 * (5 * r5);
    c += (d4 >>> 13); d4 &= 0x1fff;

    d5 = c;
    d5 += h0 * r5;
    d5 += h1 * r4;
    d5 += h2 * r3;
    d5 += h3 * r2;
    d5 += h4 * r1;
    c = (d5 >>> 13); d5 &= 0x1fff;
    d5 += h5 * r0;
    d5 += h6 * (5 * r9);
    d5 += h7 * (5 * r8);
    d5 += h8 * (5 * r7);
    d5 += h9 * (5 * r6);
    c += (d5 >>> 13); d5 &= 0x1fff;

    d6 = c;
    d6 += h0 * r6;
    d6 += h1 * r5;
    d6 += h2 * r4;
    d6 += h3 * r3;
    d6 += h4 * r2;
    c = (d6 >>> 13); d6 &= 0x1fff;
    d6 += h5 * r1;
    d6 += h6 * r0;
    d6 += h7 * (5 * r9);
    d6 += h8 * (5 * r8);
    d6 += h9 * (5 * r7);
    c += (d6 >>> 13); d6 &= 0x1fff;

    d7 = c;
    d7 += h0 * r7;
    d7 += h1 * r6;
    d7 += h2 * r5;
    d7 += h3 * r4;
    d7 += h4 * r3;
    c = (d7 >>> 13); d7 &= 0x1fff;
    d7 += h5 * r2;
    d7 += h6 * r1;
    d7 += h7 * r0;
    d7 += h8 * (5 * r9);
    d7 += h9 * (5 * r8);
    c += (d7 >>> 13); d7 &= 0x1fff;

    d8 = c;
    d8 += h0 * r8;
    d8 += h1 * r7;
    d8 += h2 * r6;
    d8 += h3 * r5;
    d8 += h4 * r4;
    c = (d8 >>> 13); d8 &= 0x1fff;
    d8 += h5 * r3;
    d8 += h6 * r2;
    d8 += h7 * r1;
    d8 += h8 * r0;
    d8 += h9 * (5 * r9);
    c += (d8 >>> 13); d8 &= 0x1fff;

    d9 = c;
    d9 += h0 * r9;
    d9 += h1 * r8;
    d9 += h2 * r7;
    d9 += h3 * r6;
    d9 += h4 * r5;
    c = (d9 >>> 13); d9 &= 0x1fff;
    d9 += h5 * r4;
    d9 += h6 * r3;
    d9 += h7 * r2;
    d9 += h8 * r1;
    d9 += h9 * r0;
    c += (d9 >>> 13); d9 &= 0x1fff;

    c = (((c << 2) + c)) | 0;
    c = (c + d0) | 0;
    d0 = c & 0x1fff;
    c = (c >>> 13);
    d1 += c;

    h0 = d0;
    h1 = d1;
    h2 = d2;
    h3 = d3;
    h4 = d4;
    h5 = d5;
    h6 = d6;
    h7 = d7;
    h8 = d8;
    h9 = d9;

    mpos += 16;
    bytes -= 16;
  }
  this.h[0] = h0;
  this.h[1] = h1;
  this.h[2] = h2;
  this.h[3] = h3;
  this.h[4] = h4;
  this.h[5] = h5;
  this.h[6] = h6;
  this.h[7] = h7;
  this.h[8] = h8;
  this.h[9] = h9;
};

poly1305.prototype.finish = function(mac, macpos) {
  var g = new Uint16Array(10);
  var c, mask, f, i;

  if (this.leftover) {
    i = this.leftover;
    this.buffer[i++] = 1;
    for (; i < 16; i++) this.buffer[i] = 0;
    this.fin = 1;
    this.blocks(this.buffer, 0, 16);
  }

  c = this.h[1] >>> 13;
  this.h[1] &= 0x1fff;
  for (i = 2; i < 10; i++) {
    this.h[i] += c;
    c = this.h[i] >>> 13;
    this.h[i] &= 0x1fff;
  }
  this.h[0] += (c * 5);
  c = this.h[0] >>> 13;
  this.h[0] &= 0x1fff;
  this.h[1] += c;
  c = this.h[1] >>> 13;
  this.h[1] &= 0x1fff;
  this.h[2] += c;

  g[0] = this.h[0] + 5;
  c = g[0] >>> 13;
  g[0] &= 0x1fff;
  for (i = 1; i < 10; i++) {
    g[i] = this.h[i] + c;
    c = g[i] >>> 13;
    g[i] &= 0x1fff;
  }
  g[9] -= (1 << 13);

  mask = (c ^ 1) - 1;
  for (i = 0; i < 10; i++) g[i] &= mask;
  mask = ~mask;
  for (i = 0; i < 10; i++) this.h[i] = (this.h[i] & mask) | g[i];

  this.h[0] = ((this.h[0]       ) | (this.h[1] << 13)                    ) & 0xffff;
  this.h[1] = ((this.h[1] >>>  3) | (this.h[2] << 10)                    ) & 0xffff;
  this.h[2] = ((this.h[2] >>>  6) | (this.h[3] <<  7)                    ) & 0xffff;
  this.h[3] = ((this.h[3] >>>  9) | (this.h[4] <<  4)                    ) & 0xffff;
  this.h[4] = ((this.h[4] >>> 12) | (this.h[5] <<  1) | (this.h[6] << 14)) & 0xffff;
  this.h[5] = ((this.h[6] >>>  2) | (this.h[7] << 11)                    ) & 0xffff;
  this.h[6] = ((this.h[7] >>>  5) | (this.h[8] <<  8)                    ) & 0xffff;
  this.h[7] = ((this.h[8] >>>  8) | (this.h[9] <<  5)                    ) & 0xffff;

  f = this.h[0] + this.pad[0];
  this.h[0] = f & 0xffff;
  for (i = 1; i < 8; i++) {
    f = (((this.h[i] + this.pad[i]) | 0) + (f >>> 16)) | 0;
    this.h[i] = f & 0xffff;
  }

  mac[macpos+ 0] = (this.h[0] >>> 0) & 0xff;
  mac[macpos+ 1] = (this.h[0] >>> 8) & 0xff;
  mac[macpos+ 2] = (this.h[1] >>> 0) & 0xff;
  mac[macpos+ 3] = (this.h[1] >>> 8) & 0xff;
  mac[macpos+ 4] = (this.h[2] >>> 0) & 0xff;
  mac[macpos+ 5] = (this.h[2] >>> 8) & 0xff;
  mac[macpos+ 6] = (this.h[3] >>> 0) & 0xff;
  mac[macpos+ 7] = (this.h[3] >>> 8) & 0xff;
  mac[macpos+ 8] = (this.h[4] >>> 0) & 0xff;
  mac[macpos+ 9] = (this.h[4] >>> 8) & 0xff;
  mac[macpos+10] = (this.h[5] >>> 0) & 0xff;
  mac[macpos+11] = (this.h[5] >>> 8) & 0xff;
  mac[macpos+12] = (this.h[6] >>> 0) & 0xff;
  mac[macpos+13] = (this.h[6] >>> 8) & 0xff;
  mac[macpos+14] = (this.h[7] >>> 0) & 0xff;
  mac[macpos+15] = (this.h[7] >>> 8) & 0xff;
};

poly1305.prototype.update = function(m, mpos, bytes) {
  var i, want;

  if (this.leftover) {
    want = (16 - this.leftover);
    if (want > bytes)
      want = bytes;
    for (i = 0; i < want; i++)
      this.buffer[this.leftover + i] = m[mpos+i];
    bytes -= want;
    mpos += want;
    this.leftover += want;
    if (this.leftover < 16)
      return;
    this.blocks(this.buffer, 0, 16);
    this.leftover = 0;
  }

  if (bytes >= 16) {
    want = bytes - (bytes % 16);
    this.blocks(m, mpos, want);
    mpos += want;
    bytes -= want;
  }

  if (bytes) {
    for (i = 0; i < bytes; i++)
      this.buffer[this.leftover + i] = m[mpos+i];
    this.leftover += bytes;
  }
};

function crypto_onetimeauth(out, outpos, m, mpos, n, k) {
  var s = new poly1305(k);
  s.update(m, mpos, n);
  s.finish(out, outpos);
  return 0;
}

function crypto_onetimeauth_verify(h, hpos, m, mpos, n, k) {
  var x = new Uint8Array(16);
  crypto_onetimeauth(x,0,m,mpos,n,k);
  return crypto_verify_16(h,hpos,x,0);
}

function crypto_secretbox(c,m,d,n,k) {
  var i;
  if (d < 32) return -1;
  crypto_stream_xor(c,0,m,0,d,n,k);
  crypto_onetimeauth(c, 16, c, 32, d - 32, c);
  for (i = 0; i < 16; i++) c[i] = 0;
  return 0;
}

function crypto_secretbox_open(m,c,d,n,k) {
  var i;
  var x = new Uint8Array(32);
  if (d < 32) return -1;
  crypto_stream(x,0,32,n,k);
  if (crypto_onetimeauth_verify(c, 16,c, 32,d - 32,x) !== 0) return -1;
  crypto_stream_xor(m,0,c,0,d,n,k);
  for (i = 0; i < 32; i++) m[i] = 0;
  return 0;
}

function set25519(r, a) {
  var i;
  for (i = 0; i < 16; i++) r[i] = a[i]|0;
}

function car25519(o) {
  var i, v, c = 1;
  for (i = 0; i < 16; i++) {
    v = o[i] + c + 65535;
    c = Math.floor(v / 65536);
    o[i] = v - c * 65536;
  }
  o[0] += c-1 + 37 * (c-1);
}

function sel25519(p, q, b) {
  var t, c = ~(b-1);
  for (var i = 0; i < 16; i++) {
    t = c & (p[i] ^ q[i]);
    p[i] ^= t;
    q[i] ^= t;
  }
}

function pack25519(o, n) {
  var i, j, b;
  var m = gf(), t = gf();
  for (i = 0; i < 16; i++) t[i] = n[i];
  car25519(t);
  car25519(t);
  car25519(t);
  for (j = 0; j < 2; j++) {
    m[0] = t[0] - 0xffed;
    for (i = 1; i < 15; i++) {
      m[i] = t[i] - 0xffff - ((m[i-1]>>16) & 1);
      m[i-1] &= 0xffff;
    }
    m[15] = t[15] - 0x7fff - ((m[14]>>16) & 1);
    b = (m[15]>>16) & 1;
    m[14] &= 0xffff;
    sel25519(t, m, 1-b);
  }
  for (i = 0; i < 16; i++) {
    o[2*i] = t[i] & 0xff;
    o[2*i+1] = t[i]>>8;
  }
}

function neq25519(a, b) {
  var c = new Uint8Array(32), d = new Uint8Array(32);
  pack25519(c, a);
  pack25519(d, b);
  return crypto_verify_32(c, 0, d, 0);
}

function par25519(a) {
  var d = new Uint8Array(32);
  pack25519(d, a);
  return d[0] & 1;
}

function unpack25519(o, n) {
  var i;
  for (i = 0; i < 16; i++) o[i] = n[2*i] + (n[2*i+1] << 8);
  o[15] &= 0x7fff;
}

function A(o, a, b) {
  for (var i = 0; i < 16; i++) o[i] = a[i] + b[i];
}

function Z(o, a, b) {
  for (var i = 0; i < 16; i++) o[i] = a[i] - b[i];
}

function M(o, a, b) {
  var v, c,
     t0 = 0,  t1 = 0,  t2 = 0,  t3 = 0,  t4 = 0,  t5 = 0,  t6 = 0,  t7 = 0,
     t8 = 0,  t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0,
    t16 = 0, t17 = 0, t18 = 0, t19 = 0, t20 = 0, t21 = 0, t22 = 0, t23 = 0,
    t24 = 0, t25 = 0, t26 = 0, t27 = 0, t28 = 0, t29 = 0, t30 = 0,
    b0 = b[0],
    b1 = b[1],
    b2 = b[2],
    b3 = b[3],
    b4 = b[4],
    b5 = b[5],
    b6 = b[6],
    b7 = b[7],
    b8 = b[8],
    b9 = b[9],
    b10 = b[10],
    b11 = b[11],
    b12 = b[12],
    b13 = b[13],
    b14 = b[14],
    b15 = b[15];

  v = a[0];
  t0 += v * b0;
  t1 += v * b1;
  t2 += v * b2;
  t3 += v * b3;
  t4 += v * b4;
  t5 += v * b5;
  t6 += v * b6;
  t7 += v * b7;
  t8 += v * b8;
  t9 += v * b9;
  t10 += v * b10;
  t11 += v * b11;
  t12 += v * b12;
  t13 += v * b13;
  t14 += v * b14;
  t15 += v * b15;
  v = a[1];
  t1 += v * b0;
  t2 += v * b1;
  t3 += v * b2;
  t4 += v * b3;
  t5 += v * b4;
  t6 += v * b5;
  t7 += v * b6;
  t8 += v * b7;
  t9 += v * b8;
  t10 += v * b9;
  t11 += v * b10;
  t12 += v * b11;
  t13 += v * b12;
  t14 += v * b13;
  t15 += v * b14;
  t16 += v * b15;
  v = a[2];
  t2 += v * b0;
  t3 += v * b1;
  t4 += v * b2;
  t5 += v * b3;
  t6 += v * b4;
  t7 += v * b5;
  t8 += v * b6;
  t9 += v * b7;
  t10 += v * b8;
  t11 += v * b9;
  t12 += v * b10;
  t13 += v * b11;
  t14 += v * b12;
  t15 += v * b13;
  t16 += v * b14;
  t17 += v * b15;
  v = a[3];
  t3 += v * b0;
  t4 += v * b1;
  t5 += v * b2;
  t6 += v * b3;
  t7 += v * b4;
  t8 += v * b5;
  t9 += v * b6;
  t10 += v * b7;
  t11 += v * b8;
  t12 += v * b9;
  t13 += v * b10;
  t14 += v * b11;
  t15 += v * b12;
  t16 += v * b13;
  t17 += v * b14;
  t18 += v * b15;
  v = a[4];
  t4 += v * b0;
  t5 += v * b1;
  t6 += v * b2;
  t7 += v * b3;
  t8 += v * b4;
  t9 += v * b5;
  t10 += v * b6;
  t11 += v * b7;
  t12 += v * b8;
  t13 += v * b9;
  t14 += v * b10;
  t15 += v * b11;
  t16 += v * b12;
  t17 += v * b13;
  t18 += v * b14;
  t19 += v * b15;
  v = a[5];
  t5 += v * b0;
  t6 += v * b1;
  t7 += v * b2;
  t8 += v * b3;
  t9 += v * b4;
  t10 += v * b5;
  t11 += v * b6;
  t12 += v * b7;
  t13 += v * b8;
  t14 += v * b9;
  t15 += v * b10;
  t16 += v * b11;
  t17 += v * b12;
  t18 += v * b13;
  t19 += v * b14;
  t20 += v * b15;
  v = a[6];
  t6 += v * b0;
  t7 += v * b1;
  t8 += v * b2;
  t9 += v * b3;
  t10 += v * b4;
  t11 += v * b5;
  t12 += v * b6;
  t13 += v * b7;
  t14 += v * b8;
  t15 += v * b9;
  t16 += v * b10;
  t17 += v * b11;
  t18 += v * b12;
  t19 += v * b13;
  t20 += v * b14;
  t21 += v * b15;
  v = a[7];
  t7 += v * b0;
  t8 += v * b1;
  t9 += v * b2;
  t10 += v * b3;
  t11 += v * b4;
  t12 += v * b5;
  t13 += v * b6;
  t14 += v * b7;
  t15 += v * b8;
  t16 += v * b9;
  t17 += v * b10;
  t18 += v * b11;
  t19 += v * b12;
  t20 += v * b13;
  t21 += v * b14;
  t22 += v * b15;
  v = a[8];
  t8 += v * b0;
  t9 += v * b1;
  t10 += v * b2;
  t11 += v * b3;
  t12 += v * b4;
  t13 += v * b5;
  t14 += v * b6;
  t15 += v * b7;
  t16 += v * b8;
  t17 += v * b9;
  t18 += v * b10;
  t19 += v * b11;
  t20 += v * b12;
  t21 += v * b13;
  t22 += v * b14;
  t23 += v * b15;
  v = a[9];
  t9 += v * b0;
  t10 += v * b1;
  t11 += v * b2;
  t12 += v * b3;
  t13 += v * b4;
  t14 += v * b5;
  t15 += v * b6;
  t16 += v * b7;
  t17 += v * b8;
  t18 += v * b9;
  t19 += v * b10;
  t20 += v * b11;
  t21 += v * b12;
  t22 += v * b13;
  t23 += v * b14;
  t24 += v * b15;
  v = a[10];
  t10 += v * b0;
  t11 += v * b1;
  t12 += v * b2;
  t13 += v * b3;
  t14 += v * b4;
  t15 += v * b5;
  t16 += v * b6;
  t17 += v * b7;
  t18 += v * b8;
  t19 += v * b9;
  t20 += v * b10;
  t21 += v * b11;
  t22 += v * b12;
  t23 += v * b13;
  t24 += v * b14;
  t25 += v * b15;
  v = a[11];
  t11 += v * b0;
  t12 += v * b1;
  t13 += v * b2;
  t14 += v * b3;
  t15 += v * b4;
  t16 += v * b5;
  t17 += v * b6;
  t18 += v * b7;
  t19 += v * b8;
  t20 += v * b9;
  t21 += v * b10;
  t22 += v * b11;
  t23 += v * b12;
  t24 += v * b13;
  t25 += v * b14;
  t26 += v * b15;
  v = a[12];
  t12 += v * b0;
  t13 += v * b1;
  t14 += v * b2;
  t15 += v * b3;
  t16 += v * b4;
  t17 += v * b5;
  t18 += v * b6;
  t19 += v * b7;
  t20 += v * b8;
  t21 += v * b9;
  t22 += v * b10;
  t23 += v * b11;
  t24 += v * b12;
  t25 += v * b13;
  t26 += v * b14;
  t27 += v * b15;
  v = a[13];
  t13 += v * b0;
  t14 += v * b1;
  t15 += v * b2;
  t16 += v * b3;
  t17 += v * b4;
  t18 += v * b5;
  t19 += v * b6;
  t20 += v * b7;
  t21 += v * b8;
  t22 += v * b9;
  t23 += v * b10;
  t24 += v * b11;
  t25 += v * b12;
  t26 += v * b13;
  t27 += v * b14;
  t28 += v * b15;
  v = a[14];
  t14 += v * b0;
  t15 += v * b1;
  t16 += v * b2;
  t17 += v * b3;
  t18 += v * b4;
  t19 += v * b5;
  t20 += v * b6;
  t21 += v * b7;
  t22 += v * b8;
  t23 += v * b9;
  t24 += v * b10;
  t25 += v * b11;
  t26 += v * b12;
  t27 += v * b13;
  t28 += v * b14;
  t29 += v * b15;
  v = a[15];
  t15 += v * b0;
  t16 += v * b1;
  t17 += v * b2;
  t18 += v * b3;
  t19 += v * b4;
  t20 += v * b5;
  t21 += v * b6;
  t22 += v * b7;
  t23 += v * b8;
  t24 += v * b9;
  t25 += v * b10;
  t26 += v * b11;
  t27 += v * b12;
  t28 += v * b13;
  t29 += v * b14;
  t30 += v * b15;

  t0  += 38 * t16;
  t1  += 38 * t17;
  t2  += 38 * t18;
  t3  += 38 * t19;
  t4  += 38 * t20;
  t5  += 38 * t21;
  t6  += 38 * t22;
  t7  += 38 * t23;
  t8  += 38 * t24;
  t9  += 38 * t25;
  t10 += 38 * t26;
  t11 += 38 * t27;
  t12 += 38 * t28;
  t13 += 38 * t29;
  t14 += 38 * t30;
  // t15 left as is

  // first car
  c = 1;
  v =  t0 + c + 65535; c = Math.floor(v / 65536);  t0 = v - c * 65536;
  v =  t1 + c + 65535; c = Math.floor(v / 65536);  t1 = v - c * 65536;
  v =  t2 + c + 65535; c = Math.floor(v / 65536);  t2 = v - c * 65536;
  v =  t3 + c + 65535; c = Math.floor(v / 65536);  t3 = v - c * 65536;
  v =  t4 + c + 65535; c = Math.floor(v / 65536);  t4 = v - c * 65536;
  v =  t5 + c + 65535; c = Math.floor(v / 65536);  t5 = v - c * 65536;
  v =  t6 + c + 65535; c = Math.floor(v / 65536);  t6 = v - c * 65536;
  v =  t7 + c + 65535; c = Math.floor(v / 65536);  t7 = v - c * 65536;
  v =  t8 + c + 65535; c = Math.floor(v / 65536);  t8 = v - c * 65536;
  v =  t9 + c + 65535; c = Math.floor(v / 65536);  t9 = v - c * 65536;
  v = t10 + c + 65535; c = Math.floor(v / 65536); t10 = v - c * 65536;
  v = t11 + c + 65535; c = Math.floor(v / 65536); t11 = v - c * 65536;
  v = t12 + c + 65535; c = Math.floor(v / 65536); t12 = v - c * 65536;
  v = t13 + c + 65535; c = Math.floor(v / 65536); t13 = v - c * 65536;
  v = t14 + c + 65535; c = Math.floor(v / 65536); t14 = v - c * 65536;
  v = t15 + c + 65535; c = Math.floor(v / 65536); t15 = v - c * 65536;
  t0 += c-1 + 37 * (c-1);

  // second car
  c = 1;
  v =  t0 + c + 65535; c = Math.floor(v / 65536);  t0 = v - c * 65536;
  v =  t1 + c + 65535; c = Math.floor(v / 65536);  t1 = v - c * 65536;
  v =  t2 + c + 65535; c = Math.floor(v / 65536);  t2 = v - c * 65536;
  v =  t3 + c + 65535; c = Math.floor(v / 65536);  t3 = v - c * 65536;
  v =  t4 + c + 65535; c = Math.floor(v / 65536);  t4 = v - c * 65536;
  v =  t5 + c + 65535; c = Math.floor(v / 65536);  t5 = v - c * 65536;
  v =  t6 + c + 65535; c = Math.floor(v / 65536);  t6 = v - c * 65536;
  v =  t7 + c + 65535; c = Math.floor(v / 65536);  t7 = v - c * 65536;
  v =  t8 + c + 65535; c = Math.floor(v / 65536);  t8 = v - c * 65536;
  v =  t9 + c + 65535; c = Math.floor(v / 65536);  t9 = v - c * 65536;
  v = t10 + c + 65535; c = Math.floor(v / 65536); t10 = v - c * 65536;
  v = t11 + c + 65535; c = Math.floor(v / 65536); t11 = v - c * 65536;
  v = t12 + c + 65535; c = Math.floor(v / 65536); t12 = v - c * 65536;
  v = t13 + c + 65535; c = Math.floor(v / 65536); t13 = v - c * 65536;
  v = t14 + c + 65535; c = Math.floor(v / 65536); t14 = v - c * 65536;
  v = t15 + c + 65535; c = Math.floor(v / 65536); t15 = v - c * 65536;
  t0 += c-1 + 37 * (c-1);

  o[ 0] = t0;
  o[ 1] = t1;
  o[ 2] = t2;
  o[ 3] = t3;
  o[ 4] = t4;
  o[ 5] = t5;
  o[ 6] = t6;
  o[ 7] = t7;
  o[ 8] = t8;
  o[ 9] = t9;
  o[10] = t10;
  o[11] = t11;
  o[12] = t12;
  o[13] = t13;
  o[14] = t14;
  o[15] = t15;
}

function S(o, a) {
  M(o, a, a);
}

function inv25519(o, i) {
  var c = gf();
  var a;
  for (a = 0; a < 16; a++) c[a] = i[a];
  for (a = 253; a >= 0; a--) {
    S(c, c);
    if(a !== 2 && a !== 4) M(c, c, i);
  }
  for (a = 0; a < 16; a++) o[a] = c[a];
}

function pow2523(o, i) {
  var c = gf();
  var a;
  for (a = 0; a < 16; a++) c[a] = i[a];
  for (a = 250; a >= 0; a--) {
      S(c, c);
      if(a !== 1) M(c, c, i);
  }
  for (a = 0; a < 16; a++) o[a] = c[a];
}

function crypto_scalarmult(q, n, p) {
  var z = new Uint8Array(32);
  var x = new Float64Array(80), r, i;
  var a = gf(), b = gf(), c = gf(),
      d = gf(), e = gf(), f = gf();
  for (i = 0; i < 31; i++) z[i] = n[i];
  z[31]=(n[31]&127)|64;
  z[0]&=248;
  unpack25519(x,p);
  for (i = 0; i < 16; i++) {
    b[i]=x[i];
    d[i]=a[i]=c[i]=0;
  }
  a[0]=d[0]=1;
  for (i=254; i>=0; --i) {
    r=(z[i>>>3]>>>(i&7))&1;
    sel25519(a,b,r);
    sel25519(c,d,r);
    A(e,a,c);
    Z(a,a,c);
    A(c,b,d);
    Z(b,b,d);
    S(d,e);
    S(f,a);
    M(a,c,a);
    M(c,b,e);
    A(e,a,c);
    Z(a,a,c);
    S(b,a);
    Z(c,d,f);
    M(a,c,_121665);
    A(a,a,d);
    M(c,c,a);
    M(a,d,f);
    M(d,b,x);
    S(b,e);
    sel25519(a,b,r);
    sel25519(c,d,r);
  }
  for (i = 0; i < 16; i++) {
    x[i+16]=a[i];
    x[i+32]=c[i];
    x[i+48]=b[i];
    x[i+64]=d[i];
  }
  var x32 = x.subarray(32);
  var x16 = x.subarray(16);
  inv25519(x32,x32);
  M(x16,x16,x32);
  pack25519(q,x16);
  return 0;
}

function crypto_scalarmult_base(q, n) {
  return crypto_scalarmult(q, n, _9);
}

function crypto_box_keypair(y, x) {
  randombytes(x, 32);
  return crypto_scalarmult_base(y, x);
}

function crypto_box_beforenm(k, y, x) {
  var s = new Uint8Array(32);
  crypto_scalarmult(s, x, y);
  return crypto_core_hsalsa20(k, _0, s, sigma);
}

var crypto_box_afternm = crypto_secretbox;
var crypto_box_open_afternm = crypto_secretbox_open;

function crypto_box(c, m, d, n, y, x) {
  var k = new Uint8Array(32);
  crypto_box_beforenm(k, y, x);
  return crypto_box_afternm(c, m, d, n, k);
}

function crypto_box_open(m, c, d, n, y, x) {
  var k = new Uint8Array(32);
  crypto_box_beforenm(k, y, x);
  return crypto_box_open_afternm(m, c, d, n, k);
}

var K = [
  0x428a2f98, 0xd728ae22, 0x71374491, 0x23ef65cd,
  0xb5c0fbcf, 0xec4d3b2f, 0xe9b5dba5, 0x8189dbbc,
  0x3956c25b, 0xf348b538, 0x59f111f1, 0xb605d019,
  0x923f82a4, 0xaf194f9b, 0xab1c5ed5, 0xda6d8118,
  0xd807aa98, 0xa3030242, 0x12835b01, 0x45706fbe,
  0x243185be, 0x4ee4b28c, 0x550c7dc3, 0xd5ffb4e2,
  0x72be5d74, 0xf27b896f, 0x80deb1fe, 0x3b1696b1,
  0x9bdc06a7, 0x25c71235, 0xc19bf174, 0xcf692694,
  0xe49b69c1, 0x9ef14ad2, 0xefbe4786, 0x384f25e3,
  0x0fc19dc6, 0x8b8cd5b5, 0x240ca1cc, 0x77ac9c65,
  0x2de92c6f, 0x592b0275, 0x4a7484aa, 0x6ea6e483,
  0x5cb0a9dc, 0xbd41fbd4, 0x76f988da, 0x831153b5,
  0x983e5152, 0xee66dfab, 0xa831c66d, 0x2db43210,
  0xb00327c8, 0x98fb213f, 0xbf597fc7, 0xbeef0ee4,
  0xc6e00bf3, 0x3da88fc2, 0xd5a79147, 0x930aa725,
  0x06ca6351, 0xe003826f, 0x14292967, 0x0a0e6e70,
  0x27b70a85, 0x46d22ffc, 0x2e1b2138, 0x5c26c926,
  0x4d2c6dfc, 0x5ac42aed, 0x53380d13, 0x9d95b3df,
  0x650a7354, 0x8baf63de, 0x766a0abb, 0x3c77b2a8,
  0x81c2c92e, 0x47edaee6, 0x92722c85, 0x1482353b,
  0xa2bfe8a1, 0x4cf10364, 0xa81a664b, 0xbc423001,
  0xc24b8b70, 0xd0f89791, 0xc76c51a3, 0x0654be30,
  0xd192e819, 0xd6ef5218, 0xd6990624, 0x5565a910,
  0xf40e3585, 0x5771202a, 0x106aa070, 0x32bbd1b8,
  0x19a4c116, 0xb8d2d0c8, 0x1e376c08, 0x5141ab53,
  0x2748774c, 0xdf8eeb99, 0x34b0bcb5, 0xe19b48a8,
  0x391c0cb3, 0xc5c95a63, 0x4ed8aa4a, 0xe3418acb,
  0x5b9cca4f, 0x7763e373, 0x682e6ff3, 0xd6b2b8a3,
  0x748f82ee, 0x5defb2fc, 0x78a5636f, 0x43172f60,
  0x84c87814, 0xa1f0ab72, 0x8cc70208, 0x1a6439ec,
  0x90befffa, 0x23631e28, 0xa4506ceb, 0xde82bde9,
  0xbef9a3f7, 0xb2c67915, 0xc67178f2, 0xe372532b,
  0xca273ece, 0xea26619c, 0xd186b8c7, 0x21c0c207,
  0xeada7dd6, 0xcde0eb1e, 0xf57d4f7f, 0xee6ed178,
  0x06f067aa, 0x72176fba, 0x0a637dc5, 0xa2c898a6,
  0x113f9804, 0xbef90dae, 0x1b710b35, 0x131c471b,
  0x28db77f5, 0x23047d84, 0x32caab7b, 0x40c72493,
  0x3c9ebe0a, 0x15c9bebc, 0x431d67c4, 0x9c100d4c,
  0x4cc5d4be, 0xcb3e42b6, 0x597f299c, 0xfc657e2a,
  0x5fcb6fab, 0x3ad6faec, 0x6c44198c, 0x4a475817
];

function crypto_hashblocks_hl(hh, hl, m, n) {
  var wh = new Int32Array(16), wl = new Int32Array(16),
      bh0, bh1, bh2, bh3, bh4, bh5, bh6, bh7,
      bl0, bl1, bl2, bl3, bl4, bl5, bl6, bl7,
      th, tl, i, j, h, l, a, b, c, d;

  var ah0 = hh[0],
      ah1 = hh[1],
      ah2 = hh[2],
      ah3 = hh[3],
      ah4 = hh[4],
      ah5 = hh[5],
      ah6 = hh[6],
      ah7 = hh[7],

      al0 = hl[0],
      al1 = hl[1],
      al2 = hl[2],
      al3 = hl[3],
      al4 = hl[4],
      al5 = hl[5],
      al6 = hl[6],
      al7 = hl[7];

  var pos = 0;
  while (n >= 128) {
    for (i = 0; i < 16; i++) {
      j = 8 * i + pos;
      wh[i] = (m[j+0] << 24) | (m[j+1] << 16) | (m[j+2] << 8) | m[j+3];
      wl[i] = (m[j+4] << 24) | (m[j+5] << 16) | (m[j+6] << 8) | m[j+7];
    }
    for (i = 0; i < 80; i++) {
      bh0 = ah0;
      bh1 = ah1;
      bh2 = ah2;
      bh3 = ah3;
      bh4 = ah4;
      bh5 = ah5;
      bh6 = ah6;
      bh7 = ah7;

      bl0 = al0;
      bl1 = al1;
      bl2 = al2;
      bl3 = al3;
      bl4 = al4;
      bl5 = al5;
      bl6 = al6;
      bl7 = al7;

      // add
      h = ah7;
      l = al7;

      a = l & 0xffff; b = l >>> 16;
      c = h & 0xffff; d = h >>> 16;

      // Sigma1
      h = ((ah4 >>> 14) | (al4 << (32-14))) ^ ((ah4 >>> 18) | (al4 << (32-18))) ^ ((al4 >>> (41-32)) | (ah4 << (32-(41-32))));
      l = ((al4 >>> 14) | (ah4 << (32-14))) ^ ((al4 >>> 18) | (ah4 << (32-18))) ^ ((ah4 >>> (41-32)) | (al4 << (32-(41-32))));

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      // Ch
      h = (ah4 & ah5) ^ (~ah4 & ah6);
      l = (al4 & al5) ^ (~al4 & al6);

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      // K
      h = K[i*2];
      l = K[i*2+1];

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      // w
      h = wh[i%16];
      l = wl[i%16];

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      b += a >>> 16;
      c += b >>> 16;
      d += c >>> 16;

      th = c & 0xffff | d << 16;
      tl = a & 0xffff | b << 16;

      // add
      h = th;
      l = tl;

      a = l & 0xffff; b = l >>> 16;
      c = h & 0xffff; d = h >>> 16;

      // Sigma0
      h = ((ah0 >>> 28) | (al0 << (32-28))) ^ ((al0 >>> (34-32)) | (ah0 << (32-(34-32)))) ^ ((al0 >>> (39-32)) | (ah0 << (32-(39-32))));
      l = ((al0 >>> 28) | (ah0 << (32-28))) ^ ((ah0 >>> (34-32)) | (al0 << (32-(34-32)))) ^ ((ah0 >>> (39-32)) | (al0 << (32-(39-32))));

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      // Maj
      h = (ah0 & ah1) ^ (ah0 & ah2) ^ (ah1 & ah2);
      l = (al0 & al1) ^ (al0 & al2) ^ (al1 & al2);

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      b += a >>> 16;
      c += b >>> 16;
      d += c >>> 16;

      bh7 = (c & 0xffff) | (d << 16);
      bl7 = (a & 0xffff) | (b << 16);

      // add
      h = bh3;
      l = bl3;

      a = l & 0xffff; b = l >>> 16;
      c = h & 0xffff; d = h >>> 16;

      h = th;
      l = tl;

      a += l & 0xffff; b += l >>> 16;
      c += h & 0xffff; d += h >>> 16;

      b += a >>> 16;
      c += b >>> 16;
      d += c >>> 16;

      bh3 = (c & 0xffff) | (d << 16);
      bl3 = (a & 0xffff) | (b << 16);

      ah1 = bh0;
      ah2 = bh1;
      ah3 = bh2;
      ah4 = bh3;
      ah5 = bh4;
      ah6 = bh5;
      ah7 = bh6;
      ah0 = bh7;

      al1 = bl0;
      al2 = bl1;
      al3 = bl2;
      al4 = bl3;
      al5 = bl4;
      al6 = bl5;
      al7 = bl6;
      al0 = bl7;

      if (i%16 === 15) {
        for (j = 0; j < 16; j++) {
          // add
          h = wh[j];
          l = wl[j];

          a = l & 0xffff; b = l >>> 16;
          c = h & 0xffff; d = h >>> 16;

          h = wh[(j+9)%16];
          l = wl[(j+9)%16];

          a += l & 0xffff; b += l >>> 16;
          c += h & 0xffff; d += h >>> 16;

          // sigma0
          th = wh[(j+1)%16];
          tl = wl[(j+1)%16];
          h = ((th >>> 1) | (tl << (32-1))) ^ ((th >>> 8) | (tl << (32-8))) ^ (th >>> 7);
          l = ((tl >>> 1) | (th << (32-1))) ^ ((tl >>> 8) | (th << (32-8))) ^ ((tl >>> 7) | (th << (32-7)));

          a += l & 0xffff; b += l >>> 16;
          c += h & 0xffff; d += h >>> 16;

          // sigma1
          th = wh[(j+14)%16];
          tl = wl[(j+14)%16];
          h = ((th >>> 19) | (tl << (32-19))) ^ ((tl >>> (61-32)) | (th << (32-(61-32)))) ^ (th >>> 6);
          l = ((tl >>> 19) | (th << (32-19))) ^ ((th >>> (61-32)) | (tl << (32-(61-32)))) ^ ((tl >>> 6) | (th << (32-6)));

          a += l & 0xffff; b += l >>> 16;
          c += h & 0xffff; d += h >>> 16;

          b += a >>> 16;
          c += b >>> 16;
          d += c >>> 16;

          wh[j] = (c & 0xffff) | (d << 16);
          wl[j] = (a & 0xffff) | (b << 16);
        }
      }
    }

    // add
    h = ah0;
    l = al0;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[0];
    l = hl[0];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[0] = ah0 = (c & 0xffff) | (d << 16);
    hl[0] = al0 = (a & 0xffff) | (b << 16);

    h = ah1;
    l = al1;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[1];
    l = hl[1];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[1] = ah1 = (c & 0xffff) | (d << 16);
    hl[1] = al1 = (a & 0xffff) | (b << 16);

    h = ah2;
    l = al2;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[2];
    l = hl[2];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[2] = ah2 = (c & 0xffff) | (d << 16);
    hl[2] = al2 = (a & 0xffff) | (b << 16);

    h = ah3;
    l = al3;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[3];
    l = hl[3];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[3] = ah3 = (c & 0xffff) | (d << 16);
    hl[3] = al3 = (a & 0xffff) | (b << 16);

    h = ah4;
    l = al4;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[4];
    l = hl[4];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[4] = ah4 = (c & 0xffff) | (d << 16);
    hl[4] = al4 = (a & 0xffff) | (b << 16);

    h = ah5;
    l = al5;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[5];
    l = hl[5];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[5] = ah5 = (c & 0xffff) | (d << 16);
    hl[5] = al5 = (a & 0xffff) | (b << 16);

    h = ah6;
    l = al6;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[6];
    l = hl[6];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[6] = ah6 = (c & 0xffff) | (d << 16);
    hl[6] = al6 = (a & 0xffff) | (b << 16);

    h = ah7;
    l = al7;

    a = l & 0xffff; b = l >>> 16;
    c = h & 0xffff; d = h >>> 16;

    h = hh[7];
    l = hl[7];

    a += l & 0xffff; b += l >>> 16;
    c += h & 0xffff; d += h >>> 16;

    b += a >>> 16;
    c += b >>> 16;
    d += c >>> 16;

    hh[7] = ah7 = (c & 0xffff) | (d << 16);
    hl[7] = al7 = (a & 0xffff) | (b << 16);

    pos += 128;
    n -= 128;
  }

  return n;
}

function crypto_hash(out, m, n) {
  var hh = new Int32Array(8),
      hl = new Int32Array(8),
      x = new Uint8Array(256),
      i, b = n;

  hh[0] = 0x6a09e667;
  hh[1] = 0xbb67ae85;
  hh[2] = 0x3c6ef372;
  hh[3] = 0xa54ff53a;
  hh[4] = 0x510e527f;
  hh[5] = 0x9b05688c;
  hh[6] = 0x1f83d9ab;
  hh[7] = 0x5be0cd19;

  hl[0] = 0xf3bcc908;
  hl[1] = 0x84caa73b;
  hl[2] = 0xfe94f82b;
  hl[3] = 0x5f1d36f1;
  hl[4] = 0xade682d1;
  hl[5] = 0x2b3e6c1f;
  hl[6] = 0xfb41bd6b;
  hl[7] = 0x137e2179;

  crypto_hashblocks_hl(hh, hl, m, n);
  n %= 128;

  for (i = 0; i < n; i++) x[i] = m[b-n+i];
  x[n] = 128;

  n = 256-128*(n<112?1:0);
  x[n-9] = 0;
  ts64(x, n-8,  (b / 0x20000000) | 0, b << 3);
  crypto_hashblocks_hl(hh, hl, x, n);

  for (i = 0; i < 8; i++) ts64(out, 8*i, hh[i], hl[i]);

  return 0;
}

function add(p, q) {
  var a = gf(), b = gf(), c = gf(),
      d = gf(), e = gf(), f = gf(),
      g = gf(), h = gf(), t = gf();

  Z(a, p[1], p[0]);
  Z(t, q[1], q[0]);
  M(a, a, t);
  A(b, p[0], p[1]);
  A(t, q[0], q[1]);
  M(b, b, t);
  M(c, p[3], q[3]);
  M(c, c, D2);
  M(d, p[2], q[2]);
  A(d, d, d);
  Z(e, b, a);
  Z(f, d, c);
  A(g, d, c);
  A(h, b, a);

  M(p[0], e, f);
  M(p[1], h, g);
  M(p[2], g, f);
  M(p[3], e, h);
}

function cswap(p, q, b) {
  var i;
  for (i = 0; i < 4; i++) {
    sel25519(p[i], q[i], b);
  }
}

function pack(r, p) {
  var tx = gf(), ty = gf(), zi = gf();
  inv25519(zi, p[2]);
  M(tx, p[0], zi);
  M(ty, p[1], zi);
  pack25519(r, ty);
  r[31] ^= par25519(tx) << 7;
}

function scalarmult(p, q, s) {
  var b, i;
  set25519(p[0], gf0);
  set25519(p[1], gf1);
  set25519(p[2], gf1);
  set25519(p[3], gf0);
  for (i = 255; i >= 0; --i) {
    b = (s[(i/8)|0] >> (i&7)) & 1;
    cswap(p, q, b);
    add(q, p);
    add(p, p);
    cswap(p, q, b);
  }
}

function scalarbase(p, s) {
  var q = [gf(), gf(), gf(), gf()];
  set25519(q[0], X);
  set25519(q[1], Y);
  set25519(q[2], gf1);
  M(q[3], X, Y);
  scalarmult(p, q, s);
}

function crypto_sign_keypair(pk, sk, seeded) {
  var d = new Uint8Array(64);
  var p = [gf(), gf(), gf(), gf()];
  var i;

  if (!seeded) randombytes(sk, 32);
  crypto_hash(d, sk, 32);
  d[0] &= 248;
  d[31] &= 127;
  d[31] |= 64;

  scalarbase(p, d);
  pack(pk, p);

  for (i = 0; i < 32; i++) sk[i+32] = pk[i];
  return 0;
}

var L = new Float64Array([0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10]);

function modL(r, x) {
  var carry, i, j, k;
  for (i = 63; i >= 32; --i) {
    carry = 0;
    for (j = i - 32, k = i - 12; j < k; ++j) {
      x[j] += carry - 16 * x[i] * L[j - (i - 32)];
      carry = Math.floor((x[j] + 128) / 256);
      x[j] -= carry * 256;
    }
    x[j] += carry;
    x[i] = 0;
  }
  carry = 0;
  for (j = 0; j < 32; j++) {
    x[j] += carry - (x[31] >> 4) * L[j];
    carry = x[j] >> 8;
    x[j] &= 255;
  }
  for (j = 0; j < 32; j++) x[j] -= carry * L[j];
  for (i = 0; i < 32; i++) {
    x[i+1] += x[i] >> 8;
    r[i] = x[i] & 255;
  }
}

function reduce(r) {
  var x = new Float64Array(64), i;
  for (i = 0; i < 64; i++) x[i] = r[i];
  for (i = 0; i < 64; i++) r[i] = 0;
  modL(r, x);
}

// Note: difference from C - smlen returned, not passed as argument.
function crypto_sign(sm, m, n, sk) {
  var d = new Uint8Array(64), h = new Uint8Array(64), r = new Uint8Array(64);
  var i, j, x = new Float64Array(64);
  var p = [gf(), gf(), gf(), gf()];

  crypto_hash(d, sk, 32);
  d[0] &= 248;
  d[31] &= 127;
  d[31] |= 64;

  var smlen = n + 64;
  for (i = 0; i < n; i++) sm[64 + i] = m[i];
  for (i = 0; i < 32; i++) sm[32 + i] = d[32 + i];

  crypto_hash(r, sm.subarray(32), n+32);
  reduce(r);
  scalarbase(p, r);
  pack(sm, p);

  for (i = 32; i < 64; i++) sm[i] = sk[i];
  crypto_hash(h, sm, n + 64);
  reduce(h);

  for (i = 0; i < 64; i++) x[i] = 0;
  for (i = 0; i < 32; i++) x[i] = r[i];
  for (i = 0; i < 32; i++) {
    for (j = 0; j < 32; j++) {
      x[i+j] += h[i] * d[j];
    }
  }

  modL(sm.subarray(32), x);
  return smlen;
}

function unpackneg(r, p) {
  var t = gf(), chk = gf(), num = gf(),
      den = gf(), den2 = gf(), den4 = gf(),
      den6 = gf();

  set25519(r[2], gf1);
  unpack25519(r[1], p);
  S(num, r[1]);
  M(den, num, D);
  Z(num, num, r[2]);
  A(den, r[2], den);

  S(den2, den);
  S(den4, den2);
  M(den6, den4, den2);
  M(t, den6, num);
  M(t, t, den);

  pow2523(t, t);
  M(t, t, num);
  M(t, t, den);
  M(t, t, den);
  M(r[0], t, den);

  S(chk, r[0]);
  M(chk, chk, den);
  if (neq25519(chk, num)) M(r[0], r[0], I);

  S(chk, r[0]);
  M(chk, chk, den);
  if (neq25519(chk, num)) return -1;

  if (par25519(r[0]) === (p[31]>>7)) Z(r[0], gf0, r[0]);

  M(r[3], r[0], r[1]);
  return 0;
}

function crypto_sign_open(m, sm, n, pk) {
  var i;
  var t = new Uint8Array(32), h = new Uint8Array(64);
  var p = [gf(), gf(), gf(), gf()],
      q = [gf(), gf(), gf(), gf()];

  if (n < 64) return -1;

  if (unpackneg(q, pk)) return -1;

  for (i = 0; i < n; i++) m[i] = sm[i];
  for (i = 0; i < 32; i++) m[i+32] = pk[i];
  crypto_hash(h, m, n);
  reduce(h);
  scalarmult(p, q, h);

  scalarbase(q, sm.subarray(32));
  add(p, q);
  pack(t, p);

  n -= 64;
  if (crypto_verify_32(sm, 0, t, 0)) {
    for (i = 0; i < n; i++) m[i] = 0;
    return -1;
  }

  for (i = 0; i < n; i++) m[i] = sm[i + 64];
  return n;
}

var crypto_secretbox_KEYBYTES = 32,
    crypto_secretbox_NONCEBYTES = 24,
    crypto_secretbox_ZEROBYTES = 32,
    crypto_secretbox_BOXZEROBYTES = 16,
    crypto_scalarmult_BYTES = 32,
    crypto_scalarmult_SCALARBYTES = 32,
    crypto_box_PUBLICKEYBYTES = 32,
    crypto_box_SECRETKEYBYTES = 32,
    crypto_box_BEFORENMBYTES = 32,
    crypto_box_NONCEBYTES = crypto_secretbox_NONCEBYTES,
    crypto_box_ZEROBYTES = crypto_secretbox_ZEROBYTES,
    crypto_box_BOXZEROBYTES = crypto_secretbox_BOXZEROBYTES,
    crypto_sign_BYTES = 64,
    crypto_sign_PUBLICKEYBYTES = 32,
    crypto_sign_SECRETKEYBYTES = 64,
    crypto_sign_SEEDBYTES = 32,
    crypto_hash_BYTES = 64;

nacl.lowlevel = {
  crypto_core_hsalsa20: crypto_core_hsalsa20,
  crypto_stream_xor: crypto_stream_xor,
  crypto_stream: crypto_stream,
  crypto_stream_salsa20_xor: crypto_stream_salsa20_xor,
  crypto_stream_salsa20: crypto_stream_salsa20,
  crypto_onetimeauth: crypto_onetimeauth,
  crypto_onetimeauth_verify: crypto_onetimeauth_verify,
  crypto_verify_16: crypto_verify_16,
  crypto_verify_32: crypto_verify_32,
  crypto_secretbox: crypto_secretbox,
  crypto_secretbox_open: crypto_secretbox_open,
  crypto_scalarmult: crypto_scalarmult,
  crypto_scalarmult_base: crypto_scalarmult_base,
  crypto_box_beforenm: crypto_box_beforenm,
  crypto_box_afternm: crypto_box_afternm,
  crypto_box: crypto_box,
  crypto_box_open: crypto_box_open,
  crypto_box_keypair: crypto_box_keypair,
  crypto_hash: crypto_hash,
  crypto_sign: crypto_sign,
  crypto_sign_keypair: crypto_sign_keypair,
  crypto_sign_open: crypto_sign_open,

  crypto_secretbox_KEYBYTES: crypto_secretbox_KEYBYTES,
  crypto_secretbox_NONCEBYTES: crypto_secretbox_NONCEBYTES,
  crypto_secretbox_ZEROBYTES: crypto_secretbox_ZEROBYTES,
  crypto_secretbox_BOXZEROBYTES: crypto_secretbox_BOXZEROBYTES,
  crypto_scalarmult_BYTES: crypto_scalarmult_BYTES,
  crypto_scalarmult_SCALARBYTES: crypto_scalarmult_SCALARBYTES,
  crypto_box_PUBLICKEYBYTES: crypto_box_PUBLICKEYBYTES,
  crypto_box_SECRETKEYBYTES: crypto_box_SECRETKEYBYTES,
  crypto_box_BEFORENMBYTES: crypto_box_BEFORENMBYTES,
  crypto_box_NONCEBYTES: crypto_box_NONCEBYTES,
  crypto_box_ZEROBYTES: crypto_box_ZEROBYTES,
  crypto_box_BOXZEROBYTES: crypto_box_BOXZEROBYTES,
  crypto_sign_BYTES: crypto_sign_BYTES,
  crypto_sign_PUBLICKEYBYTES: crypto_sign_PUBLICKEYBYTES,
  crypto_sign_SECRETKEYBYTES: crypto_sign_SECRETKEYBYTES,
  crypto_sign_SEEDBYTES: crypto_sign_SEEDBYTES,
  crypto_hash_BYTES: crypto_hash_BYTES,

  gf: gf,
  D: D,
  L: L,
  pack25519: pack25519,
  unpack25519: unpack25519,
  M: M,
  A: A,
  S: S,
  Z: Z,
  pow2523: pow2523,
  add: add,
  set25519: set25519,
  modL: modL,
  scalarmult: scalarmult,
  scalarbase: scalarbase,
};

/* High-level API */

function checkLengths(k, n) {
  if (k.length !== crypto_secretbox_KEYBYTES) throw new Error('bad key size');
  if (n.length !== crypto_secretbox_NONCEBYTES) throw new Error('bad nonce size');
}

function checkBoxLengths(pk, sk) {
  if (pk.length !== crypto_box_PUBLICKEYBYTES) throw new Error('bad public key size');
  if (sk.length !== crypto_box_SECRETKEYBYTES) throw new Error('bad secret key size');
}

function checkArrayTypes() {
  for (var i = 0; i < arguments.length; i++) {
    if (!(arguments[i] instanceof Uint8Array))
      throw new TypeError('unexpected type, use Uint8Array');
  }
}

function cleanup(arr) {
  for (var i = 0; i < arr.length; i++) arr[i] = 0;
}

nacl.randomBytes = function(n) {
  var b = new Uint8Array(n);
  randombytes(b, n);
  return b;
};

nacl.secretbox = function(msg, nonce, key) {
  checkArrayTypes(msg, nonce, key);
  checkLengths(key, nonce);
  var m = new Uint8Array(crypto_secretbox_ZEROBYTES + msg.length);
  var c = new Uint8Array(m.length);
  for (var i = 0; i < msg.length; i++) m[i+crypto_secretbox_ZEROBYTES] = msg[i];
  crypto_secretbox(c, m, m.length, nonce, key);
  return c.subarray(crypto_secretbox_BOXZEROBYTES);
};

nacl.secretbox.open = function(box, nonce, key) {
  checkArrayTypes(box, nonce, key);
  checkLengths(key, nonce);
  var c = new Uint8Array(crypto_secretbox_BOXZEROBYTES + box.length);
  var m = new Uint8Array(c.length);
  for (var i = 0; i < box.length; i++) c[i+crypto_secretbox_BOXZEROBYTES] = box[i];
  if (c.length < 32) return null;
  if (crypto_secretbox_open(m, c, c.length, nonce, key) !== 0) return null;
  return m.subarray(crypto_secretbox_ZEROBYTES);
};

nacl.secretbox.keyLength = crypto_secretbox_KEYBYTES;
nacl.secretbox.nonceLength = crypto_secretbox_NONCEBYTES;
nacl.secretbox.overheadLength = crypto_secretbox_BOXZEROBYTES;

nacl.scalarMult = function(n, p) {
  checkArrayTypes(n, p);
  if (n.length !== crypto_scalarmult_SCALARBYTES) throw new Error('bad n size');
  if (p.length !== crypto_scalarmult_BYTES) throw new Error('bad p size');
  var q = new Uint8Array(crypto_scalarmult_BYTES);
  crypto_scalarmult(q, n, p);
  return q;
};

nacl.scalarMult.base = function(n) {
  checkArrayTypes(n);
  if (n.length !== crypto_scalarmult_SCALARBYTES) throw new Error('bad n size');
  var q = new Uint8Array(crypto_scalarmult_BYTES);
  crypto_scalarmult_base(q, n);
  return q;
};

nacl.scalarMult.scalarLength = crypto_scalarmult_SCALARBYTES;
nacl.scalarMult.groupElementLength = crypto_scalarmult_BYTES;

nacl.box = function(msg, nonce, publicKey, secretKey) {
  var k = nacl.box.before(publicKey, secretKey);
  return nacl.secretbox(msg, nonce, k);
};

nacl.box.before = function(publicKey, secretKey) {
  checkArrayTypes(publicKey, secretKey);
  checkBoxLengths(publicKey, secretKey);
  var k = new Uint8Array(crypto_box_BEFORENMBYTES);
  crypto_box_beforenm(k, publicKey, secretKey);
  return k;
};

nacl.box.after = nacl.secretbox;

nacl.box.open = function(msg, nonce, publicKey, secretKey) {
  var k = nacl.box.before(publicKey, secretKey);
  return nacl.secretbox.open(msg, nonce, k);
};

nacl.box.open.after = nacl.secretbox.open;

nacl.box.keyPair = function() {
  var pk = new Uint8Array(crypto_box_PUBLICKEYBYTES);
  var sk = new Uint8Array(crypto_box_SECRETKEYBYTES);
  crypto_box_keypair(pk, sk);
  return {publicKey: pk, secretKey: sk};
};

nacl.box.keyPair.fromSecretKey = function(secretKey) {
  checkArrayTypes(secretKey);
  if (secretKey.length !== crypto_box_SECRETKEYBYTES)
    throw new Error('bad secret key size');
  var pk = new Uint8Array(crypto_box_PUBLICKEYBYTES);
  crypto_scalarmult_base(pk, secretKey);
  return {publicKey: pk, secretKey: new Uint8Array(secretKey)};
};

nacl.box.publicKeyLength = crypto_box_PUBLICKEYBYTES;
nacl.box.secretKeyLength = crypto_box_SECRETKEYBYTES;
nacl.box.sharedKeyLength = crypto_box_BEFORENMBYTES;
nacl.box.nonceLength = crypto_box_NONCEBYTES;
nacl.box.overheadLength = nacl.secretbox.overheadLength;

nacl.sign = function(msg, secretKey) {
  checkArrayTypes(msg, secretKey);
  if (secretKey.length !== crypto_sign_SECRETKEYBYTES)
    throw new Error('bad secret key size');
  var signedMsg = new Uint8Array(crypto_sign_BYTES+msg.length);
  crypto_sign(signedMsg, msg, msg.length, secretKey);
  return signedMsg;
};

nacl.sign.open = function(signedMsg, publicKey) {
  checkArrayTypes(signedMsg, publicKey);
  if (publicKey.length !== crypto_sign_PUBLICKEYBYTES)
    throw new Error('bad public key size');
  var tmp = new Uint8Array(signedMsg.length);
  var mlen = crypto_sign_open(tmp, signedMsg, signedMsg.length, publicKey);
  if (mlen < 0) return null;
  var m = new Uint8Array(mlen);
  for (var i = 0; i < m.length; i++) m[i] = tmp[i];
  return m;
};

nacl.sign.detached = function(msg, secretKey) {
  var signedMsg = nacl.sign(msg, secretKey);
  var sig = new Uint8Array(crypto_sign_BYTES);
  for (var i = 0; i < sig.length; i++) sig[i] = signedMsg[i];
  return sig;
};

nacl.sign.detached.verify = function(msg, sig, publicKey) {
  checkArrayTypes(msg, sig, publicKey);
  if (sig.length !== crypto_sign_BYTES)
    throw new Error('bad signature size');
  if (publicKey.length !== crypto_sign_PUBLICKEYBYTES)
    throw new Error('bad public key size');
  var sm = new Uint8Array(crypto_sign_BYTES + msg.length);
  var m = new Uint8Array(crypto_sign_BYTES + msg.length);
  var i;
  for (i = 0; i < crypto_sign_BYTES; i++) sm[i] = sig[i];
  for (i = 0; i < msg.length; i++) sm[i+crypto_sign_BYTES] = msg[i];
  return (crypto_sign_open(m, sm, sm.length, publicKey) >= 0);
};

nacl.sign.keyPair = function() {
  var pk = new Uint8Array(crypto_sign_PUBLICKEYBYTES);
  var sk = new Uint8Array(crypto_sign_SECRETKEYBYTES);
  crypto_sign_keypair(pk, sk);
  return {publicKey: pk, secretKey: sk};
};

nacl.sign.keyPair.fromSecretKey = function(secretKey) {
  checkArrayTypes(secretKey);
  if (secretKey.length !== crypto_sign_SECRETKEYBYTES)
    throw new Error('bad secret key size');
  var pk = new Uint8Array(crypto_sign_PUBLICKEYBYTES);
  for (var i = 0; i < pk.length; i++) pk[i] = secretKey[32+i];
  return {publicKey: pk, secretKey: new Uint8Array(secretKey)};
};

nacl.sign.keyPair.fromSeed = function(seed) {
  checkArrayTypes(seed);
  if (seed.length !== crypto_sign_SEEDBYTES)
    throw new Error('bad seed size');
  var pk = new Uint8Array(crypto_sign_PUBLICKEYBYTES);
  var sk = new Uint8Array(crypto_sign_SECRETKEYBYTES);
  for (var i = 0; i < 32; i++) sk[i] = seed[i];
  crypto_sign_keypair(pk, sk, true);
  return {publicKey: pk, secretKey: sk};
};

nacl.sign.publicKeyLength = crypto_sign_PUBLICKEYBYTES;
nacl.sign.secretKeyLength = crypto_sign_SECRETKEYBYTES;
nacl.sign.seedLength = crypto_sign_SEEDBYTES;
nacl.sign.signatureLength = crypto_sign_BYTES;

nacl.hash = function(msg) {
  checkArrayTypes(msg);
  var h = new Uint8Array(crypto_hash_BYTES);
  crypto_hash(h, msg, msg.length);
  return h;
};

nacl.hash.hashLength = crypto_hash_BYTES;

nacl.verify = function(x, y) {
  checkArrayTypes(x, y);
  // Zero length arguments are considered not equal.
  if (x.length === 0 || y.length === 0) return false;
  if (x.length !== y.length) return false;
  return (vn(x, 0, y, 0, x.length) === 0) ? true : false;
};

nacl.setPRNG = function(fn) {
  randombytes = fn;
};

(function() {
  // Initialize PRNG if environment provides CSPRNG.
  // If not, methods calling randombytes will throw.
  var crypto = typeof self !== 'undefined' ? (self.crypto || self.msCrypto) : null;
  if (crypto && crypto.getRandomValues) {
    // Browsers.
    var QUOTA = 65536;
    nacl.setPRNG(function(x, n) {
      var i, v = new Uint8Array(n);
      for (i = 0; i < n; i += QUOTA) {
        crypto.getRandomValues(v.subarray(i, i + Math.min(n - i, QUOTA)));
      }
      for (i = 0; i < n; i++) x[i] = v[i];
      cleanup(v);
    });
  } else if (true) {
    // Node.js.
    crypto = __webpack_require__(/*! crypto */ "?dba7");
    if (crypto && crypto.randomBytes) {
      nacl.setPRNG(function(x, n) {
        var i, v = crypto.randomBytes(n);
        for (i = 0; i < n; i++) x[i] = v[i];
        cleanup(v);
      });
    }
  }
})();

})( true && module.exports ? module.exports : (self.nacl = self.nacl || {}));


/***/ }),

/***/ "./src/crypto.js":
/*!***********************!*\
  !*** ./src/crypto.js ***!
  \***********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   decryptPayload: () => (/* binding */ decryptPayload),
/* harmony export */   encryptPayload: () => (/* binding */ encryptPayload),
/* harmony export */   generateEncryptionKeypair: () => (/* binding */ generateEncryptionKeypair),
/* harmony export */   publicKeyFromHex: () => (/* binding */ publicKeyFromHex),
/* harmony export */   publicKeyToHex: () => (/* binding */ publicKeyToHex)
/* harmony export */ });
/**
 * Cryptographic utilities for hypha-rpc end-to-end encrypted service calls.
 *
 * Uses tweetnacl (libsodium-compatible) for:
 * - Curve25519 ECDH + XSalsa20-Poly1305 authenticated encryption (crypto_box)
 *
 * All output is interoperable with the Python PyNaCl package.
 *
 * tweetnacl is an optional dependency. If it is not installed, the hex-conversion
 * helpers still work, but encryption/decryption functions will throw a clear error.
 *
 * @module crypto
 */

let _nacl = null;

function _requireCrypto() {
  if (_nacl) return _nacl;
  try {
    // Use require() for synchronous loading; works in both bundled and Node.js contexts
    _nacl = __webpack_require__(/*! tweetnacl */ "./node_modules/tweetnacl/nacl-fast.js");
    // Handle ES module default export
    if (_nacl.default) _nacl = _nacl.default;
    return _nacl;
  } catch (e) {
    throw new Error(
      "The 'tweetnacl' package is required for encryption. " +
        "Install it with: npm install tweetnacl",
    );
  }
}

/**
 * Convert a 32-byte public key to a 64-character hex string.
 * @param {Uint8Array} publicKeyBytes - 32-byte raw public key.
 * @returns {string} 64-character hex string.
 */
function publicKeyToHex(publicKeyBytes) {
  return Array.from(publicKeyBytes, (b) => b.toString(16).padStart(2, "0")).join(
    "",
  );
}

/**
 * Convert a 64-character hex string to a 32-byte public key.
 * @param {string} hexString - 64-character hex string.
 * @returns {Uint8Array} 32-byte raw public key.
 */
function publicKeyFromHex(hexString) {
  const bytes = new Uint8Array(hexString.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hexString.substr(i * 2, 2), 16);
  }
  return bytes;
}

// --- Curve25519 + XSalsa20-Poly1305 Encryption (crypto_box) ---

/**
 * Generate a Curve25519 keypair for authenticated encryption.
 * @returns {Promise<{privateKey: Uint8Array, publicKey: Uint8Array}>}
 *   Raw 32-byte private key and 32-byte public key.
 */
async function generateEncryptionKeypair() {
  const nacl = _requireCrypto();
  const kp = nacl.box.keyPair();
  return { privateKey: kp.secretKey, publicKey: kp.publicKey };
}

/**
 * Encrypt plaintext using crypto_box (Curve25519 + XSalsa20-Poly1305).
 * @param {Uint8Array} myPrivateBytes - 32-byte raw Curve25519 private key.
 * @param {Uint8Array} theirPublicBytes - 32-byte raw Curve25519 public key.
 * @param {Uint8Array} plaintext - Data to encrypt.
 * @returns {Promise<{nonce: Uint8Array, ciphertext: Uint8Array}>}
 *   nonce is 24 bytes; ciphertext includes 16-byte Poly1305 auth tag.
 */
async function encryptPayload(myPrivateBytes, theirPublicBytes, plaintext) {
  const nacl = _requireCrypto();
  const nonce = nacl.randomBytes(nacl.box.nonceLength); // 24 bytes
  const ciphertext = nacl.box(plaintext, nonce, theirPublicBytes, myPrivateBytes);
  return { nonce, ciphertext };
}

/**
 * Decrypt ciphertext using crypto_box (Curve25519 + XSalsa20-Poly1305).
 * @param {Uint8Array} myPrivateBytes - 32-byte raw Curve25519 private key.
 * @param {Uint8Array} theirPublicBytes - 32-byte raw Curve25519 public key.
 * @param {Uint8Array} nonce - 24-byte nonce used during encryption.
 * @param {Uint8Array} ciphertext - Data to decrypt (includes 16-byte Poly1305 auth tag).
 * @returns {Promise<Uint8Array>} Decrypted plaintext.
 * @throws {Error} If decryption/authentication fails.
 */
async function decryptPayload(myPrivateBytes, theirPublicBytes, nonce, ciphertext) {
  const nacl = _requireCrypto();
  const plaintext = nacl.box.open(ciphertext, nonce, theirPublicBytes, myPrivateBytes);
  if (plaintext === null) {
    throw new Error("Decryption failed");
  }
  return plaintext;
}


/***/ }),

/***/ "./src/http-client.js":
/*!****************************!*\
  !*** ./src/http-client.js ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   HTTPStreamingRPCConnection: () => (/* binding */ HTTPStreamingRPCConnection),
/* harmony export */   _connectToServerHTTP: () => (/* binding */ _connectToServerHTTP),
/* harmony export */   connectToServerHTTP: () => (/* binding */ connectToServerHTTP),
/* harmony export */   getRemoteServiceHTTP: () => (/* binding */ getRemoteServiceHTTP),
/* harmony export */   normalizeServerUrl: () => (/* binding */ normalizeServerUrl)
/* harmony export */ });
/* harmony import */ var _rpc_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./rpc.js */ "./src/rpc.js");
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs");
/**
 * HTTP Streaming RPC Client for Hypha.
 *
 * This module provides HTTP-based RPC transport as an alternative to WebSocket.
 * It uses:
 * - HTTP GET with streaming (msgpack) for server-to-client messages
 * - HTTP POST for client-to-server messages
 *
 * This is more resilient to network issues than WebSocket because:
 * 1. Each POST request is independent (stateless)
 * 2. GET stream can be easily reconnected
 * 3. Works through more proxies and firewalls
 *
 * ## Performance Optimizations
 *
 * Modern browsers automatically provide optimal HTTP performance:
 *
 * ### Automatic HTTP/2 Support
 * - Browsers negotiate HTTP/2 when server supports it
 * - Multiplexing: Multiple requests over single TCP connection
 * - Header compression: HPACK reduces overhead
 * - Server push: Pre-emptive resource delivery
 *
 * ### Connection Pooling
 * - Browsers maintain connection pools per origin
 * - Automatic keep-alive for HTTP/1.1
 * - Connection reuse reduces latency
 * - No manual configuration needed
 *
 * ### Fetch API Optimizations
 * - `keepalive: true` flag ensures connection reuse
 * - Streaming responses with backpressure handling
 * - Efficient binary data transfer (ArrayBuffer/Uint8Array)
 *
 * ### Server-Side Configuration
 * For optimal performance, ensure server has:
 * - Keep-alive timeout: 300s (matches typical browser defaults) ✓ CONFIGURED
 * - Fast compression: gzip level 1 (2-5x faster than level 5) ✓ CONFIGURED
 * - Uvicorn connection limits optimized ✓ CONFIGURED
 *
 * ### HTTP/2 Support
 * - Uvicorn does NOT natively support HTTP/2 (as of 2026)
 * - In production, use nginx/Caddy/ALB as reverse proxy for HTTP/2
 * - Reverse proxy handles HTTP/2 ↔ HTTP/1.1 translation
 * - Browsers automatically use HTTP/2 when reverse proxy supports it
 * - Current HTTP/1.1 implementation is already optimal
 *
 * ### Performance Results
 * With properly configured server, HTTP transport achieves:
 * - 10-12 MB/s throughput for large payloads (4-15 MB)
 * - 2-3x faster than before optimization
 * - 3-28x faster than WebSocket for data transfer
 * - 31% improvement in connection reuse efficiency
 */






const MAX_RETRY = 1000000;

/**
 * Extract workspace from a JWT token's scope claim.
 *
 * JWT tokens issued by the Hypha server contain a `scope` field with
 * entries like "wid:<workspace>" indicating the current workspace.
 * This function decodes the JWT payload (base64url-encoded JSON) to
 * extract the workspace without requiring any crypto library.
 *
 * @param {string} token - JWT token string
 * @returns {string|null} Workspace name or null if not found
 */
function extractWorkspaceFromToken(token) {
  if (!token) return null;
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    // Decode the payload (second part) from base64url
    let payload = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    // Pad to multiple of 4
    while (payload.length % 4) payload += "=";
    const decoded = JSON.parse(atob(payload));
    const scope = decoded.scope;
    if (typeof scope !== "string") return null;
    // Look for "wid:<workspace>" in the scope string
    const match = scope.match(/(?:^|\s)wid:(\S+)/);
    return match ? match[1] : null;
  } catch {
    return null;
  }
}

/**
 * HTTP Streaming RPC Connection.
 *
 * Uses HTTP GET with streaming for receiving messages and HTTP POST for sending messages.
 * Uses msgpack binary format with length-prefixed frames for efficient binary data support.
 */
class HTTPStreamingRPCConnection {
  /**
   * Initialize HTTP streaming connection.
   *
   * @param {string} server_url - The server URL (http:// or https://)
   * @param {string} client_id - Unique client identifier
   * @param {string} workspace - Target workspace (optional)
   * @param {string} token - Authentication token (optional)
   * @param {string} reconnection_token - Token for reconnection (optional)
   * @param {number} timeout - Request timeout in seconds (default: 60)
   * @param {number} token_refresh_interval - Interval for token refresh (default: 2 hours)
   */
  constructor(
    server_url,
    client_id,
    workspace = null,
    token = null,
    reconnection_token = null,
    timeout = 60,
    token_refresh_interval = 2 * 60 * 60,
  ) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(server_url && client_id, "server_url and client_id are required");
    this._server_url = server_url.replace(/\/$/, "");
    this._client_id = client_id;
    this._workspace = workspace;
    this._token = token;
    this._reconnection_token = reconnection_token;
    this._timeout = timeout;
    this._token_refresh_interval = token_refresh_interval;

    // If no workspace specified but token provided, extract workspace from token
    if (!this._workspace && this._token) {
      this._workspace = extractWorkspaceFromToken(this._token);
    }

    this._handle_message = null;
    this._handle_disconnected = null;
    this._handle_connected = null;

    this._closed = false;
    this._enable_reconnect = false;
    this._is_reconnection = false;
    this.connection_info = null;
    this.manager_id = null;

    this._abort_controller = null;
    this._refresh_token_task = null;
  }

  /**
   * Register message handler.
   */
  on_message(handler) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(handler, "handler is required");
    this._handle_message = handler;
  }

  /**
   * Register disconnection handler.
   */
  on_disconnected(handler) {
    this._handle_disconnected = handler;
  }

  /**
   * Register connection handler.
   */
  on_connected(handler) {
    this._handle_connected = handler;
  }

  /**
   * Get HTTP headers with authentication.
   *
   * @param {boolean} for_stream - If true, set Accept header for msgpack stream
   * @returns {Object} Headers object
   */
  _get_headers(for_stream = false) {
    const headers = {
      "Content-Type": "application/msgpack",
    };
    if (for_stream) {
      headers["Accept"] = "application/x-msgpack-stream";
    }
    if (this._token) {
      headers["Authorization"] = `Bearer ${this._token}`;
    }
    return headers;
  }

  /**
   * Open the streaming connection.
   */
  async open() {
    console.info(`Opening HTTP streaming connection to ${this._server_url}`);

    // Build stream URL
    // At this point, _workspace should be set from: user config, or token extraction.
    // For anonymous users (no token, no workspace), fall back to "public" which the
    // server redirects to the user's own workspace via connection_info.
    const ws = this._workspace || "public";
    const stream_url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;

    // Start streaming in background
    this._startStreamLoop(stream_url);

    // Wait for connection info (first message)
    const start = Date.now();
    while (this.connection_info === null) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      if (Date.now() - start > this._timeout * 1000) {
        throw new Error("Timeout waiting for connection info");
      }
      if (this._closed) {
        throw new Error("Connection closed during setup");
      }
    }

    this.manager_id = this.connection_info.manager_id;
    if (this._workspace) {
      const actual_ws = this.connection_info.workspace;
      if (actual_ws !== this._workspace) {
        throw new Error(
          `Connected to wrong workspace: ${actual_ws}, expected: ${this._workspace}`,
        );
      }
    }
    this._workspace = this.connection_info.workspace;

    if (this.connection_info.reconnection_token) {
      this._reconnection_token = this.connection_info.reconnection_token;
    }

    // Adjust token refresh interval based on server's token lifetime
    if (this.connection_info.reconnection_token_life_time) {
      const token_life_time = this.connection_info.reconnection_token_life_time;
      if (this._token_refresh_interval > token_life_time / 1.5) {
        console.warn(
          `Token refresh interval (${this._token_refresh_interval}s) is too long, ` +
            `adjusting to ${(token_life_time / 1.5).toFixed(0)}s based on token lifetime`,
        );
        this._token_refresh_interval = token_life_time / 1.5;
      }
    }

    console.info(
      `HTTP streaming connected to workspace: ${this._workspace}, ` +
        `manager_id: ${this.manager_id}`,
    );

    // Start token refresh
    if (this._token_refresh_interval > 0) {
      this._startTokenRefresh();
    }

    if (this._handle_connected) {
      await this._handle_connected(this.connection_info);
    }

    return this.connection_info;
  }

  /**
   * Start periodic token refresh via POST.
   */
  _startTokenRefresh() {
    // Clear existing refresh if any
    if (this._refresh_token_task) {
      clearInterval(this._refresh_token_task);
    }

    // Initial delay of 2s, then periodic refresh
    setTimeout(() => {
      this._sendRefreshToken();
      this._refresh_token_task = setInterval(() => {
        this._sendRefreshToken();
      }, this._token_refresh_interval * 1000);
    }, 2000);
  }

  /**
   * Send a token refresh request via POST.
   */
  async _sendRefreshToken() {
    if (this._closed) return;
    try {
      // After open(), _workspace is always set from connection_info
      const ws = this._workspace;
      if (!ws) return;
      const url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;
      const body = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.encode)({ type: "refresh_token" });
      const response = await fetch(url, {
        method: "POST",
        headers: this._get_headers(false),
        body: body,
      });
      if (response.ok) {
        console.debug("Token refresh requested successfully");
      } else {
        console.warn(`Token refresh request failed: ${response.status}`);
      }
    } catch (e) {
      console.warn(`Failed to send refresh token request: ${e.message}`);
    }
  }

  /**
   * Start the streaming loop.
   *
   * OPTIMIZATION: Modern browsers automatically:
   * - Negotiate HTTP/2 when server supports it
   * - Use connection pooling for multiple requests to same origin
   * - Handle keep-alive for persistent connections
   * - Stream responses efficiently with backpressure handling
   */
  async _startStreamLoop(url) {
    this._enable_reconnect = true;
    this._closed = false;
    let retry = 0;
    this._is_reconnection = false;

    while (!this._closed && retry < MAX_RETRY) {
      try {
        // Update URL with current workspace (set from connection_info after initial open)
        const ws = this._workspace || "public";
        let stream_url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;
        if (this._reconnection_token) {
          stream_url += `&reconnection_token=${encodeURIComponent(this._reconnection_token)}`;
        }

        // OPTIMIZATION: Browser fetch automatically streams responses
        // and negotiates HTTP/2 when available for better performance
        this._abort_controller = new AbortController();
        const response = await fetch(stream_url, {
          method: "GET",
          headers: this._get_headers(true),
          signal: this._abort_controller.signal,
        });

        if (!response.ok) {
          const error_text = await response.text();
          throw new Error(
            `Stream failed with status ${response.status}: ${error_text}`,
          );
        }

        retry = 0; // Reset retry counter on successful connection

        // Process binary msgpack stream with 4-byte length prefix
        await this._processMsgpackStream(response);
      } catch (error) {
        if (this._closed) break;
        console.error(`Connection error: ${error.message}`);

        if (!this._enable_reconnect) {
          break;
        }
      }

      // After the first connection attempt, all subsequent ones are reconnections
      this._is_reconnection = true;

      // Reconnection logic
      if (!this._closed && this._enable_reconnect) {
        retry += 1;
        // Exponential backoff with max 60 seconds
        const delay = Math.min(Math.pow(2, Math.min(retry, 6)), 60);
        console.warn(
          `Stream disconnected, reconnecting in ${delay.toFixed(1)}s (attempt ${retry})`,
        );
        await new Promise((resolve) => setTimeout(resolve, delay * 1000));
      } else {
        break;
      }
    }

    if (!this._closed && this._handle_disconnected) {
      this._handle_disconnected("Stream ended");
    }
  }

  /**
   * Check if frame data is a control message and decode it.
   *
   * Control messages vs RPC messages:
   * - Control messages: Single msgpack object with "type" field (connection_info, ping, etc.)
   * - RPC messages: May contain multiple concatenated msgpack objects (main message + extra data)
   *
   * We only need to decode the first object to check if it's a control message.
   * RPC messages are passed as raw bytes to the handler.
   *
   * @param {Uint8Array} frame_data - The msgpack frame data
   * @returns {Object|null} Decoded control message or null
   */
  _tryDecodeControlMessage(frame_data) {
    try {
      // Decode the first msgpack object in the frame
      const decoded = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.decode)(frame_data);

      // Control messages are simple objects with a "type" field
      if (typeof decoded === "object" && decoded !== null && decoded.type) {
        const controlTypes = [
          "connection_info",
          "ping",
          "pong",
          "reconnection_token",
          "error",
        ];
        if (controlTypes.includes(decoded.type)) {
          return decoded;
        }
      }

      // Not a control message
      return null;
    } catch {
      // Decode failed - this is an RPC message
      return null;
    }
  }

  /**
   * Process msgpack stream with 4-byte length prefix.
   */
  async _processMsgpackStream(response) {
    const reader = response.body.getReader();
    // Growing buffer to avoid O(n^2) re-allocation on every chunk
    let buffer = new Uint8Array(4096);
    let bufferLen = 0;

    while (!this._closed) {
      const { done, value } = await reader.read();

      if (done) break;

      // Grow buffer if needed (double size until it fits)
      const needed = bufferLen + value.length;
      if (needed > buffer.length) {
        let newSize = buffer.length;
        while (newSize < needed) newSize *= 2;
        const grown = new Uint8Array(newSize);
        grown.set(buffer.subarray(0, bufferLen));
        buffer = grown;
      }
      buffer.set(value, bufferLen);
      bufferLen += value.length;

      // Process complete frames from buffer
      let offset = 0;
      while (bufferLen - offset >= 4) {
        // Read 4-byte length prefix (big-endian)
        const length =
          (buffer[offset] << 24) |
          (buffer[offset + 1] << 16) |
          (buffer[offset + 2] << 8) |
          buffer[offset + 3];

        if (bufferLen - offset < 4 + length) {
          // Incomplete frame, wait for more data
          break;
        }

        // Extract the frame (slice creates a copy, which is needed since buffer is reused)
        const frame_data = buffer.slice(offset + 4, offset + 4 + length);
        offset += 4 + length;

        // Try to decode as control message first
        const controlMsg = this._tryDecodeControlMessage(frame_data);
        if (controlMsg) {
          const msg_type = controlMsg.type;
          if (msg_type === "connection_info") {
            this.connection_info = controlMsg;
            // On reconnection, update state and notify RPC layer.
            // Run as a non-blocking task so the stream can continue
            // processing incoming RPC responses (the reconnection
            // handler sends RPC calls that need stream responses).
            if (this._is_reconnection) {
              this._handleReconnection(controlMsg).catch((err) => {
                console.error(`Reconnection handling failed: ${err.message}`);
              });
            }
            continue;
          } else if (msg_type === "ping" || msg_type === "pong") {
            continue;
          } else if (msg_type === "reconnection_token") {
            this._reconnection_token = controlMsg.reconnection_token;
            continue;
          } else if (msg_type === "error") {
            console.error(`Server error: ${controlMsg.message}`);
            continue;
          }
        }

        // For RPC messages (or unrecognized control messages), pass raw frame data to handler
        if (this._handle_message) {
          try {
            await this._handle_message(frame_data);
          } catch (error) {
            console.error(`Error in message handler: ${error.message}`);
          }
        }
      }
      // Compact: shift remaining data to the front of the buffer
      if (offset > 0) {
        const remaining = bufferLen - offset;
        if (remaining > 0) {
          buffer.copyWithin(0, offset, bufferLen);
        }
        bufferLen = remaining;
      }
    }
  }

  /**
   * Handle reconnection: update state and notify RPC layer.
   */
  async _handleReconnection(connection_info) {
    this.manager_id = connection_info.manager_id;
    this._workspace = connection_info.workspace;

    if (connection_info.reconnection_token) {
      this._reconnection_token = connection_info.reconnection_token;
    }

    // Adjust token refresh interval if needed
    if (connection_info.reconnection_token_life_time) {
      const token_life_time = connection_info.reconnection_token_life_time;
      if (this._token_refresh_interval > token_life_time / 1.5) {
        this._token_refresh_interval = token_life_time / 1.5;
      }
    }

    console.warn(
      `Stream reconnected to workspace: ${this._workspace}, ` +
        `manager_id: ${this.manager_id}`,
    );

    // Notify RPC layer so it can re-register services
    if (this._handle_connected) {
      await this._handle_connected(this.connection_info);
    }

    // Wait a short time for services to be re-registered
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  /**
   * Send a message to the server via HTTP POST.
   *
   * OPTIMIZATION: Uses keepalive flag for connection reuse.
   * Modern browsers automatically:
   * - Use HTTP/2 when available (multiplexing, header compression)
   * - Manage connection pooling with HTTP/1.1 keep-alive
   * - Reuse connections for same-origin requests
   */
  async emit_message(data) {
    if (this._closed) {
      throw new Error("Connection is closed");
    }

    // Build POST URL - workspace is always set after open() from connection_info
    const ws = this._workspace;
    if (!ws) {
      throw new Error("Workspace not set - connection not established");
    }
    let post_url = `${this._server_url}/${ws}/rpc?client_id=${this._client_id}`;

    // Ensure data is Uint8Array
    const body = data instanceof Uint8Array ? data : new Uint8Array(data);

    // Retry logic to handle transient issues such as load balancer
    // routing POST requests to a different server instance than the GET stream
    const maxRetries = 3;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        // Note: keepalive has a 64KB body size limit in browsers, so only use
        // it for small payloads. For large payloads, skip keepalive.
        const useKeepalive = body.length < 60000;
        const response = await fetch(post_url, {
          method: "POST",
          headers: this._get_headers(false),
          body: body,
          ...(useKeepalive && { keepalive: true }),
        });

        if (!response.ok) {
          const error_text = await response.text();
          // Retry on 400 errors that indicate the server doesn't recognize
          // our stream (e.g., load balancer routed to a different instance)
          if (response.status === 400 && attempt < maxRetries - 1) {
            console.warn(
              `POST failed (attempt ${attempt + 1}/${maxRetries}): ${error_text}, retrying...`,
            );
            await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
            continue;
          }
          throw new Error(
            `POST failed with status ${response.status}: ${error_text}`,
          );
        }

        return true;
      } catch (error) {
        if (attempt < maxRetries - 1 && !this._closed) {
          console.warn(
            `Failed to send message (attempt ${attempt + 1}/${maxRetries}): ${error.message}, retrying...`,
          );
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
        } else {
          console.error(`Failed to send message: ${error.message}`);
          throw error;
        }
      }
    }
  }

  /**
   * Set reconnection flag.
   */
  set_reconnection(value) {
    this._enable_reconnect = value;
  }

  /**
   * Close the connection.
   */
  async disconnect(reason = "client disconnect") {
    if (this._closed) return;

    this._closed = true;

    // Clear token refresh interval
    if (this._refresh_token_task) {
      clearInterval(this._refresh_token_task);
      this._refresh_token_task = null;
    }

    // Abort any active stream fetch to release the connection immediately
    if (this._abort_controller) {
      this._abort_controller.abort();
      this._abort_controller = null;
    }

    if (this._handle_disconnected) {
      this._handle_disconnected(reason);
    }
  }
}

/**
 * Normalize server URL for HTTP transport.
 */
function normalizeServerUrl(server_url) {
  if (!server_url) {
    throw new Error("server_url is required");
  }

  // Convert ws:// to http://
  if (server_url.startsWith("ws://")) {
    server_url = server_url.replace("ws://", "http://");
  } else if (server_url.startsWith("wss://")) {
    server_url = server_url.replace("wss://", "https://");
  }

  // Remove /ws suffix if present (WebSocket endpoint)
  if (server_url.endsWith("/ws")) {
    server_url = server_url.slice(0, -3);
  }

  return server_url.replace(/\/$/, "");
}

/**
 * Internal function to establish HTTP streaming connection.
 */
async function _connectToServerHTTP(config) {
  let clientId = config.clientId || config.client_id;
  if (!clientId) {
    clientId = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();
  }

  const server_url = normalizeServerUrl(config.serverUrl || config.server_url);

  const connection = new HTTPStreamingRPCConnection(
    server_url,
    clientId,
    config.workspace,
    config.token,
    config.reconnection_token,
    config.method_timeout || 30,
    config.token_refresh_interval || 2 * 60 * 60,
  );

  const connection_info = await connection.open();
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(connection_info, "Failed to connect to server");

  await new Promise((resolve) => setTimeout(resolve, 100));

  const workspace = connection_info.workspace;

  const rpc = new _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC(connection, {
    client_id: clientId,
    workspace,
    default_context: { connection_type: "http_streaming" },
    name: config.name,
    method_timeout: config.method_timeout,
    app_id: config.app_id,
    server_base_url: connection_info.public_base_url,
    encryption: config.encryption || false,
    encryption_private_key: config.encryption_private_key || null,
    encryption_public_key: config.encryption_public_key || null,
  });

  await rpc.waitFor("services_registered", config.method_timeout || 120);

  const wm = await rpc.get_manager_service({
    timeout: config.method_timeout || 30,
    case_conversion: "camel",
  });
  wm.rpc = rpc;

  // Add standard methods
  wm.disconnect = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.disconnect.bind(rpc), {
    name: "disconnect",
    description: "Disconnect from server",
    parameters: { properties: {}, type: "object" },
  });

  wm.registerService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.register_service.bind(rpc), {
    name: "registerService",
    description: "Register a service",
    parameters: {
      properties: {
        service: { description: "Service to register", type: "object" },
      },
      required: ["service"],
      type: "object",
    },
  });

  const _getService = wm.getService;
  wm.getService = async (query, config = {}) => {
    return await _getService(query, config);
  };
  if (_getService.__schema__) {
    wm.getService.__schema__ = _getService.__schema__;
  }

  async function serve() {
    await new Promise(() => {}); // Wait forever
  }

  wm.serve = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(serve, {
    name: "serve",
    description: "Run event loop forever",
    parameters: { type: "object", properties: {} },
  });

  if (connection_info) {
    wm.config = Object.assign(wm.config || {}, connection_info);
  }

  // Handle force-exit from manager
  if (connection.manager_id) {
    rpc.on("force-exit", async (message) => {
      if (message.from === "*/" + connection.manager_id) {
        console.info(`Disconnecting from server: ${message.reason}`);
        await rpc.disconnect();
      }
    });
  }

  return wm;
}

/**
 * Connect to server using HTTP streaming transport.
 *
 * This is an alternative to WebSocket connection that's more resilient
 * to network issues.
 *
 * @param {Object} config - Configuration object
 * @returns {Promise<Object>} Connected workspace manager
 */
async function connectToServerHTTP(config = {}) {
  return await _connectToServerHTTP(config);
}

/**
 * Get a remote service using HTTP transport.
 */
async function getRemoteServiceHTTP(serviceUri, config = {}) {
  const { serverUrl, workspace, clientId, serviceId, appId } =
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.parseServiceUrl)(serviceUri);
  const fullServiceId = `${workspace}/${clientId}:${serviceId}@${appId}`;

  if (config.serverUrl) {
    if (config.serverUrl !== serverUrl) {
      throw new Error("server_url mismatch");
    }
  }
  config.serverUrl = serverUrl;

  const server = await connectToServerHTTP(config);
  return await server.getService(fullServiceId);
}


/***/ }),

/***/ "./src/rpc.js":
/*!********************!*\
  !*** ./src/rpc.js ***!
  \********************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   API_VERSION: () => (/* binding */ API_VERSION),
/* harmony export */   RPC: () => (/* binding */ RPC),
/* harmony export */   _applyEncryptionKeyToService: () => (/* binding */ _applyEncryptionKeyToService)
/* harmony export */ });
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs");
/* harmony import */ var _msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @msgpack/msgpack */ "./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs");
/* harmony import */ var _crypto_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./crypto.js */ "./src/crypto.js");
/**
 * Contains the RPC object used both by the application
 * site, and by each plugin
 */






/**
 * Apply an out-of-band encryption public key to all remote methods in a service.
 * @param {Object} svc - The decoded service object.
 * @param {Uint8Array} encPubBytes - 32-byte raw X25519 public key.
 */
function _applyEncryptionKeyToService(svc, encPubBytes) {
  if (svc && typeof svc === "object" && !ArrayBuffer.isView(svc)) {
    for (const key of Object.keys(svc)) {
      const value = svc[key];
      if (typeof value === "function" && value.__rpc_object__) {
        value.__rpc_object__._renc_pub = encPubBytes;
      } else if (
        value &&
        typeof value === "object" &&
        !ArrayBuffer.isView(value)
      ) {
        _applyEncryptionKeyToService(value, encPubBytes);
      }
    }
  }
}



const API_VERSION = 3;
const CHUNK_SIZE = 1024 * 256;
const CONCURRENCY_LIMIT = 30;

const ArrayBufferView = Object.getPrototypeOf(
  Object.getPrototypeOf(new Uint8Array()),
).constructor;

/**
 * Check if a value is a primitive type that needs no encoding/decoding.
 */
function _isPrimitive(v) {
  if (v === null || v === undefined) return true;
  const t = typeof v;
  return t === "number" || t === "string" || t === "boolean";
}

/**
 * Check if an array (and nested arrays/objects) contains only primitives.
 */
function _allPrimitivesArray(arr) {
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (_isPrimitive(v)) continue;
    if (v instanceof Uint8Array) continue;
    if (Array.isArray(v)) {
      if (!_allPrimitivesArray(v)) return false;
      continue;
    }
    if (v && v.constructor === Object) {
      if ("_rtype" in v) return false;
      if (!_allPrimitivesObject(v)) return false;
      continue;
    }
    return false;
  }
  return true;
}

/**
 * Check if an object (and nested objects/arrays) contains only primitives.
 */
function _allPrimitivesObject(obj) {
  const values = Object.values(obj);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (_isPrimitive(v)) continue;
    if (v instanceof Uint8Array) continue;
    if (Array.isArray(v)) {
      if (!_allPrimitivesArray(v)) return false;
      continue;
    }
    if (v && v.constructor === Object) {
      if ("_rtype" in v) return false;
      if (!_allPrimitivesObject(v)) return false;
      continue;
    }
    return false;
  }
  return true;
}

function _appendBuffer(buffer1, buffer2) {
  const tmp = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
  tmp.set(new Uint8Array(buffer1), 0);
  tmp.set(new Uint8Array(buffer2), buffer1.byteLength);
  return tmp.buffer;
}

/**
 * Wrap a promise with a timeout.
 * @param {Promise} promise - The promise to wrap.
 * @param {number} timeoutMs - The timeout in milliseconds.
 * @param {string} message - Optional error message for timeout.
 * @returns {Promise} - The wrapped promise that will reject on timeout.
 */
function withTimeout(promise, timeoutMs, message = "Operation timed out") {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error(`TimeoutError: ${message}`));
    }, timeoutMs);

    promise
      .then((result) => {
        clearTimeout(timeoutId);
        resolve(result);
      })
      .catch((error) => {
        clearTimeout(timeoutId);
        reject(error);
      });
  });
}

function indexObject(obj, is) {
  if (!is) throw new Error("undefined index");
  if (typeof is === "string") return indexObject(obj, is.split("."));
  else if (is.length === 0) return obj;
  else return indexObject(obj[is[0]], is.slice(1));
}

// Simple fallback schema generation - no docstring parsing for JS

function _get_schema(obj, name = null, skipContext = false) {
  if (Array.isArray(obj)) {
    return obj.map((v, i) => _get_schema(v, null, skipContext));
  } else if (typeof obj === "object" && obj !== null) {
    let schema = {};
    for (let k in obj) {
      schema[k] = _get_schema(obj[k], k, skipContext);
    }
    return schema;
  } else if (typeof obj === "function") {
    if (obj.__schema__) {
      const schema = JSON.parse(JSON.stringify(obj.__schema__));
      if (name) {
        schema.name = name;
        obj.__schema__.name = name;
      }
      if (skipContext) {
        if (schema.parameters && schema.parameters.properties) {
          delete schema.parameters.properties["context"];
        }
        if (schema.parameters && schema.parameters.required) {
          const contextIndex = schema.parameters.required.indexOf("context");
          if (contextIndex > -1) {
            schema.parameters.required.splice(contextIndex, 1);
          }
        }
      }
      return { type: "function", function: schema };
    } else {
      // Simple fallback for JavaScript - just return basic function schema with name
      const funcName = name || obj.name || "function";
      return {
        type: "function",
        function: {
          name: funcName,
        },
      };
    }
  } else if (typeof obj === "number") {
    return { type: "number" };
  } else if (typeof obj === "string") {
    return { type: "string" };
  } else if (typeof obj === "boolean") {
    return { type: "boolean" };
  } else if (obj === null) {
    return { type: "null" };
  } else {
    return {};
  }
}

function _annotate_service(service, serviceTypeInfo) {
  function validateKeys(serviceDict, schemaDict, path = "root") {
    // Validate that all keys in schemaDict exist in serviceDict
    for (let key in schemaDict) {
      if (!serviceDict.hasOwnProperty(key)) {
        throw new Error(`Missing key '${key}' in service at path '${path}'`);
      }
    }

    // Check for any unexpected keys in serviceDict
    for (let key in serviceDict) {
      if (key !== "type" && !schemaDict.hasOwnProperty(key)) {
        throw new Error(`Unexpected key '${key}' in service at path '${path}'`);
      }
    }
  }

  function annotateRecursive(newService, schemaInfo, path = "root") {
    if (typeof newService === "object" && !Array.isArray(newService)) {
      validateKeys(newService, schemaInfo, path);
      for (let k in newService) {
        let v = newService[k];
        let newPath = `${path}.${k}`;
        if (typeof v === "object" && !Array.isArray(v)) {
          annotateRecursive(v, schemaInfo[k], newPath);
        } else if (typeof v === "function") {
          if (schemaInfo.hasOwnProperty(k)) {
            newService[k] = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_1__.schemaFunction)(v, {
              name: schemaInfo[k]["name"],
              description: schemaInfo[k].description || "",
              parameters: schemaInfo[k]["parameters"],
            });
          } else {
            throw new Error(
              `Missing schema for function '${k}' at path '${newPath}'`,
            );
          }
        }
      }
    } else if (Array.isArray(newService)) {
      if (newService.length !== schemaInfo.length) {
        throw new Error(`Length mismatch at path '${path}'`);
      }
      newService.forEach((v, i) => {
        let newPath = `${path}[${i}]`;
        if (typeof v === "object" && !Array.isArray(v)) {
          annotateRecursive(v, schemaInfo[i], newPath);
        } else if (typeof v === "function") {
          if (schemaInfo.hasOwnProperty(i)) {
            newService[i] = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_1__.schemaFunction)(v, {
              name: schemaInfo[i]["name"],
              description: schemaInfo[i].description || "",
              parameters: schemaInfo[i]["parameters"],
            });
          } else {
            throw new Error(
              `Missing schema for function at index ${i} in path '${newPath}'`,
            );
          }
        }
      });
    }
  }

  validateKeys(service, serviceTypeInfo["definition"]);
  annotateRecursive(service, serviceTypeInfo["definition"]);
  return service;
}

function getFunctionInfo(func) {
  const funcString = func.toString();

  // Extract function name
  const nameMatch = funcString.match(/function\s*(\w*)/);
  const name = (nameMatch && nameMatch[1]) || "";

  // Extract function parameters, excluding comments
  const paramsMatch = funcString.match(/\(([^)]*)\)/);
  let params = "";
  if (paramsMatch) {
    params = paramsMatch[1]
      .split(",")
      .map((p) =>
        p
          .replace(/\/\*.*?\*\//g, "") // Remove block comments
          .replace(/\/\/.*$/g, ""),
      ) // Remove line comments
      .filter((p) => p.trim().length > 0) // Remove empty strings after removing comments
      .map((p) => p.trim()) // Trim remaining whitespace
      .join(", ");
  }

  // Extract function docstring (block comment)
  let docMatch = funcString.match(/\)\s*\{\s*\/\*([\s\S]*?)\*\//);
  const docstringBlock = (docMatch && docMatch[1].trim()) || "";

  // Extract function docstring (line comment)
  docMatch = funcString.match(/\)\s*\{\s*(\/\/[\s\S]*?)\n\s*[^\s\/]/);
  const docstringLine =
    (docMatch &&
      docMatch[1]
        .split("\n")
        .map((s) => s.replace(/^\/\/\s*/, "").trim())
        .join("\n")) ||
    "";

  const docstring = docstringBlock || docstringLine;
  return (
    name &&
    params.length > 0 && {
      name: name,
      sig: params,
      doc: docstring,
    }
  );
}

function concatArrayBuffers(buffers) {
  var buffersLengths = buffers.map(function (b) {
      return b.byteLength;
    }),
    totalBufferlength = buffersLengths.reduce(function (p, c) {
      return p + c;
    }, 0),
    unit8Arr = new Uint8Array(totalBufferlength);
  buffersLengths.reduce(function (p, c, i) {
    unit8Arr.set(new Uint8Array(buffers[i]), p);
    return p + c;
  }, 0);
  return unit8Arr.buffer;
}

class Timer {
  constructor(timeout, callback, args, label) {
    this._timeout = timeout;
    this._callback = callback;
    this._args = args;
    this._label = label || "timer";
    this._task = null;
    this.started = false;
  }

  start() {
    if (this.started) {
      this.reset();
    } else {
      this._task = setTimeout(() => {
        this._callback.apply(this, this._args);
      }, this._timeout * 1000);
      this.started = true;
    }
  }

  clear() {
    if (this._task && this.started) {
      clearTimeout(this._task);
      this._task = null;
      this.started = false;
    } else {
      console.warn(`Clearing a timer (${this._label}) which is not started`);
    }
  }

  reset() {
    if (this._task) {
      clearTimeout(this._task);
    }
    this._task = setTimeout(() => {
      this._callback.apply(this, this._args);
    }, this._timeout * 1000);
    this.started = true;
  }
}

class RemoteService extends Object {}

/**
 * RPC object represents a single site in the
 * communication protocol between the application and the plugin
 *
 * @param {Object} connection a special object allowing to send
 * and receive messages from the opposite site (basically it
 * should only provide send() and onMessage() methods)
 */
class RPC extends _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.MessageEmitter {
  constructor(
    connection,
    {
      client_id = null,
      default_context = null,
      name = null,
      codecs = null,
      method_timeout = null,
      max_message_buffer_size = 0,
      debug = false,
      workspace = null,
      silent = false,
      app_id = null,
      server_base_url = null,
      long_message_chunk_size = null,
      encryption = false,
      encryption_private_key = null,
      encryption_public_key = null,
    },
  ) {
    super(debug);
    this._codecs = codecs || {};
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(client_id && typeof client_id === "string");
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(client_id, "client_id is required");
    this._client_id = client_id;
    this._name = name;
    this._app_id = app_id || "*";
    this._local_workspace = workspace;
    this._silent = silent;
    this.default_context = default_context || {};
    this._method_annotations = new WeakMap();
    this._max_message_buffer_size = max_message_buffer_size;
    this._chunk_store = {};
    this._method_timeout = method_timeout || 30;
    this._server_base_url = server_base_url;
    this._long_message_chunk_size = long_message_chunk_size || CHUNK_SIZE;

    // make sure there is an execute function
    this._services = {};
    this._object_store = {
      services: this._services,
    };
    // Index: target_id -> Set of top-level session keys for fast cleanup
    this._targetIdIndex = {};

    // Encryption support (X25519 ECDH + AES-256-GCM)
    this._encryption_enabled = false;
    this._encryption_private_key = null;
    this._encryption_public_key = null;
    this._encryption_ready = null;
    if (encryption) {
      if (encryption_private_key && encryption_public_key) {
        this._encryption_private_key = encryption_private_key;
        this._encryption_public_key = encryption_public_key;
        this._encryption_enabled = true;
        this._encryption_ready = Promise.resolve();
      } else {
        this._encryption_ready = (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.generateEncryptionKeypair)().then((keyPair) => {
          this._encryption_private_key = keyPair.privateKey;
          this._encryption_public_key = keyPair.publicKey;
          this._encryption_enabled = true;
        });
      }
    }

    // Track background tasks for proper cleanup
    this._background_tasks = new Set();

    // Periodic session sweep for stale sessions with no activity.
    // Default: 10 minutes (matching Python).
    this._sessionMaxAge = 10 * 60 * 1000;
    this._sessionSweepInterval = setInterval(() => {
      this._sweepStaleSessions();
    }, this._sessionMaxAge / 2);

    // Set up global unhandled promise rejection handler for RPC-related errors
    // Use a class-level reference counter so the handler is added once and removed
    // only when the last RPC instance is closed.
    this._unhandledRejectionHandler = (event) => {
      const reason = event.reason;
      if (reason && typeof reason === "object") {
        const reasonStr = reason.toString();
        if (
          reasonStr.includes("Method not found") ||
          reasonStr.includes("Session not found") ||
          reasonStr.includes("Method expired") ||
          reasonStr.includes("Connection is closed") ||
          reasonStr.includes("Client disconnected") ||
          reasonStr.includes("RPC connection closed")
        ) {
          console.debug(
            "Ignoring expected disconnection/method error:",
            reason,
          );
          event.preventDefault();
          return;
        }
      }
      console.warn("Unhandled RPC promise rejection:", reason);
    };

    this._unhandledRejectionNodeHandler = null;

    if (!RPC._rejectionHandlerCount) {
      RPC._rejectionHandlerCount = 0;
    }
    RPC._rejectionHandlerCount++;

    if (RPC._rejectionHandlerCount === 1) {
      if (typeof window !== "undefined") {
        window.addEventListener(
          "unhandledrejection",
          this._unhandledRejectionHandler,
        );
      } else if (typeof process !== "undefined") {
        this._unhandledRejectionNodeHandler = (reason, promise) => {
          this._unhandledRejectionHandler({
            reason,
            promise,
            preventDefault: () => {},
          });
        };
        process.on("unhandledRejection", this._unhandledRejectionNodeHandler);
      }
    }

    if (connection) {
      this.add_service({
        id: "built-in",
        type: "built-in",
        name: `Built-in services for ${this._local_workspace}/${this._client_id}`,
        config: {
          require_context: true,
          visibility: "public",
          api_version: API_VERSION,
        },
        ping: this._ping.bind(this),
        get_service: this.get_local_service.bind(this),
        message_cache: {
          create: this._create_message.bind(this),
          append: this._append_message.bind(this),
          set: this._set_message.bind(this),
          process: this._process_message.bind(this),
          remove: this._remove_message.bind(this),
        },
      });
      this._boundHandleMethod = this._handle_method.bind(this);
      this._boundHandleError = console.error;
      this.on("method", this._boundHandleMethod);
      this.on("error", this._boundHandleError);

      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(connection.emit_message && connection.on_message);
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        connection.manager_id !== undefined,
        "Connection must have manager_id",
      );
      this._emit_message = connection.emit_message.bind(connection);
      connection.on_message(this._on_message.bind(this));
      this._connection = connection;
      const onConnected = async (connectionInfo) => {
        if (!this._silent && this._connection.manager_id) {
          console.debug("Connection established, reporting services...");
          try {
            // Retry getting manager service with exponential backoff
            const manager = await this.get_manager_service({
              timeout: 20,
              case_conversion: "camel",
            });
            const services = Object.values(this._services);
            const servicesCount = services.length;
            let registeredCount = 0;
            const failedServices = [];

            // Use timeout for service registration to prevent hanging
            // _method_timeout is in seconds, withTimeout expects milliseconds
            const serviceRegistrationTimeout = (this._method_timeout || 30) * 1000;

            for (let service of services) {
              try {
                const serviceInfo = this._extract_service_info(service);
                await withTimeout(
                  manager.registerService(serviceInfo),
                  serviceRegistrationTimeout,
                  `Timeout registering service ${service.id || "unknown"}`,
                );
                registeredCount++;
                console.debug(
                  `Successfully registered service: ${service.id || "unknown"}`,
                );
              } catch (serviceError) {
                failedServices.push(service.id || "unknown");
                if (
                  serviceError.message &&
                  serviceError.message.includes("TimeoutError")
                ) {
                  console.error(
                    `Timeout registering service ${service.id || "unknown"}`,
                  );
                } else {
                  console.error(
                    `Failed to register service ${service.id || "unknown"}: ${serviceError}`,
                  );
                }
              }
            }

            if (registeredCount === servicesCount) {
              console.info(
                `Successfully registered all ${registeredCount} services with the server`,
              );
            } else {
              console.warn(
                `Only registered ${registeredCount} out of ${servicesCount} services with the server. Failed services: ${failedServices.join(", ")}`,
              );
            }

            // Fire event with registration status
            this._fire("services_registered", {
              total: servicesCount,
              registered: registeredCount,
              failed: failedServices,
            });

            // Subscribe to client_disconnected events if the manager supports it
            try {
              if (
                manager.subscribe &&
                typeof manager.subscribe === "function"
              ) {
                console.debug("Subscribing to client_disconnected events");

                const handleClientDisconnected = async (event) => {
                  // The client ID is in event.data.id based on the event structure
                  const clientId = event.data?.id || event.client;
                  const workspace = event.data?.workspace;
                  if (clientId && workspace) {
                    // Construct the full client path with workspace prefix
                    const fullClientId = `${workspace}/${clientId}`;
                    console.debug(
                      `Client ${fullClientId} disconnected, cleaning up sessions`,
                    );
                    await this._handleClientDisconnected(fullClientId);
                  } else if (clientId) {
                    console.debug(
                      `Client ${clientId} disconnected, cleaning up sessions`,
                    );
                    await this._handleClientDisconnected(clientId);
                  }
                };

                // Subscribe to the event topic first with timeout
                this._clientDisconnectedSubscription = await withTimeout(
                  manager.subscribe(["client_disconnected"]),
                  serviceRegistrationTimeout,
                  "Timeout subscribing to client_disconnected events",
                );

                // Then register the local event handler
                this.on("client_disconnected", handleClientDisconnected);

                console.debug(
                  "Successfully subscribed to client_disconnected events",
                );
              } else {
                console.debug(
                  "Manager does not support subscribe method, skipping client_disconnected handling",
                );
                this._clientDisconnectedSubscription = null;
              }
            } catch (subscribeError) {
              console.debug(
                `Failed to subscribe to client_disconnected events: ${subscribeError}`,
              );
              this._clientDisconnectedSubscription = null;
            }
          } catch (managerError) {
            console.error(
              `Failed to get manager service for registering services: ${managerError}`,
            );
            // Fire event with error status
            this._fire("services_registration_failed", {
              error: managerError.toString(),
              total_services: Object.keys(this._services).length,
            });
          }
        } else {
          // console.debug("Connection established", connectionInfo);
        }
        if (connectionInfo) {
          if (connectionInfo.public_base_url) {
            this._server_base_url = connectionInfo.public_base_url;
          }
          this._fire("connected", connectionInfo);
        }
      };
      connection.on_connected(onConnected);

      // Register disconnect handler to reject all pending RPC calls
      // This ensures no remote function call hangs forever when the connection drops
      if (typeof connection.on_disconnected === "function") {
        connection.on_disconnected((reason) => {
          // If reconnection is enabled, don't reject pending calls immediately.
          // The timeout mechanism will handle them if reconnection fails,
          // allowing calls to succeed after a successful reconnection.
          if (connection._enable_reconnect) {
            console.info(
              `Connection lost (${reason}), reconnection enabled - pending calls will be handled by timeout`,
            );
            return;
          }
          console.warn(
            `Connection lost (${reason}), rejecting all pending RPC calls`,
          );
          this._rejectPendingCalls(
            `Connection lost: ${reason || "unknown reason"}`,
          );
        });
      }

      onConnected();
    } else {
      this._emit_message = function () {
        console.log("No connection to emit message");
      };
    }
  }

  /**
   * Return this client's X25519 public key as a 64-char hex string.
   * Returns null if encryption is not enabled.
   */
  getPublicKey() {
    if (!this._encryption_enabled || !this._encryption_public_key) return null;
    return (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.publicKeyToHex)(this._encryption_public_key);
  }

  register_codec(config) {
    if (!config["name"] || (!config["encoder"] && !config["decoder"])) {
      throw new Error(
        "Invalid codec format, please make sure you provide a name, type, encoder and decoder.",
      );
    } else {
      if (config.type) {
        for (let k of Object.keys(this._codecs)) {
          if (this._codecs[k].type === config.type || k === config.name) {
            delete this._codecs[k];
            console.warn("Remove duplicated codec: " + k);
          }
        }
      }
      this._codecs[config["name"]] = config;
    }
  }

  async _ping(msg, context) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(msg == "ping");
    return "pong";
  }

  async ping(client_id, timeout) {
    let method = this._generate_remote_method({
      _rserver: this._server_base_url,
      _rtarget: client_id,
      _rmethod: "services.built-in.ping",
      _rpromise: true,
      _rdoc: "Ping a remote client",
    });
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)((await method("ping", timeout)) == "pong");
  }

  _create_message(key, heartbeat, overwrite, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }

    if (!this._object_store["message_cache"]) {
      this._object_store["message_cache"] = {};
    }

    // Evict stale cache entries (older than 5 minutes) and enforce size limit
    const cache = this._object_store["message_cache"];
    const MAX_CACHE_SIZE = 256;
    const MAX_CACHE_AGE = 5 * 60 * 1000;
    const cacheKeys = Object.keys(cache);
    if (cacheKeys.length >= MAX_CACHE_SIZE) {
      const now = Date.now();
      for (const k of cacheKeys) {
        const entry = cache[k];
        if (
          entry &&
          entry._cache_created_at &&
          now - entry._cache_created_at > MAX_CACHE_AGE
        ) {
          delete cache[k];
        }
      }
      // If still over limit, evict oldest entries
      const remaining = Object.keys(cache);
      if (remaining.length >= MAX_CACHE_SIZE) {
        remaining
          .sort(
            (a, b) =>
              (cache[a]._cache_created_at || 0) -
              (cache[b]._cache_created_at || 0),
          )
          .slice(0, remaining.length - MAX_CACHE_SIZE + 1)
          .forEach((k) => delete cache[k]);
      }
    }

    if (!overwrite && cache[key]) {
      throw new Error(
        `Message with the same key (${key}) already exists in the cache store, please use overwrite=true or remove it first.`,
      );
    }
    cache[key] = [];
    cache[key]._cache_created_at = Date.now();
  }

  _append_message(key, data, heartbeat, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }
    const cache = this._object_store["message_cache"];
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(data instanceof ArrayBufferView);
    cache[key].push(data);
  }

  _set_message(key, index, data, heartbeat, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }
    const cache = this._object_store["message_cache"];
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(data instanceof ArrayBufferView);
    cache[key][index] = data;
  }

  _remove_message(key, context) {
    const cache = this._object_store["message_cache"];
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    delete cache[key];
  }

  _process_message(key, heartbeat, context) {
    if (heartbeat) {
      if (!this._object_store[key]) {
        throw new Error(`session does not exist anymore: ${key}`);
      }
      this._object_store[key]["timer"].reset();
    }
    const cache = this._object_store["message_cache"];
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(!!context, "Context is required");
    if (!cache[key]) {
      throw new Error(`Message with key ${key} does not exists.`);
    }
    cache[key] = concatArrayBuffers(cache[key]);
    // console.debug(`Processing message ${key} (bytes=${cache[key].byteLength})`);
    let unpacker = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.decodeMulti)(cache[key]);
    const { done, value } = unpacker.next();
    const main = value;
    // Make sure the fields are from trusted source
    Object.assign(main, {
      from: context.from,
      to: context.to,
      ws: context.ws,
      user: context.user,
    });
    main["ctx"] = JSON.parse(JSON.stringify(main));
    Object.assign(main["ctx"], this.default_context);
    if (!done) {
      let extra = unpacker.next();
      Object.assign(main, extra.value);
    }
    this._fire(main["type"], main);
    // console.debug(
    //   this._client_id,
    //   `Processed message ${key} (bytes=${cache[key].byteLength})`,
    // );
    delete cache[key];
  }

  _on_message(message) {
    if (typeof message === "string") {
      const main = JSON.parse(message);
      // Add trusted context to the method call
      main["ctx"] = Object.assign({}, main, this.default_context);
      this._fire(main["type"], main);
    } else if (message instanceof ArrayBuffer || ArrayBuffer.isView(message)) {
      // Handle both ArrayBuffer (WebSocket) and Uint8Array/ArrayBufferView (HTTP transport)
      let unpacker = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.decodeMulti)(message);
      const { done, value } = unpacker.next();
      const main = value;
      // Add trusted context to the method call
      main["ctx"] = Object.assign({}, main, this.default_context);
      if (!done) {
        let extra = unpacker.next();
        Object.assign(main, extra.value);
      }
      this._fire(main["type"], main);
    } else if (typeof message === "object") {
      // Add trusted context to the method call
      message["ctx"] = Object.assign({}, message, this.default_context);
      this._fire(message["type"], message);
    } else {
      throw new Error("Invalid message format");
    }
  }

  reset() {
    this._removeRejectionHandler();
    this._event_handlers = {};
    this._services = {};
  }

  _removeRejectionHandler() {
    if (RPC._rejectionHandlerCount && RPC._rejectionHandlerCount > 0) {
      RPC._rejectionHandlerCount--;
      if (RPC._rejectionHandlerCount === 0) {
        if (typeof window !== "undefined" && this._unhandledRejectionHandler) {
          window.removeEventListener(
            "unhandledrejection",
            this._unhandledRejectionHandler,
          );
        } else if (
          typeof process !== "undefined" &&
          this._unhandledRejectionNodeHandler
        ) {
          process.removeListener(
            "unhandledRejection",
            this._unhandledRejectionNodeHandler,
          );
        }
      }
    }
    this._unhandledRejectionHandler = null;
    this._unhandledRejectionNodeHandler = null;
  }

  close() {
    // Clean up all pending sessions (rejects promises, clears timers/heartbeats, deletes sessions)
    this._cleanupOnDisconnect();

    // Remove method and error event listeners
    if (this._boundHandleMethod) {
      this.off("method", this._boundHandleMethod);
      this._boundHandleMethod = null;
    }
    if (this._boundHandleError) {
      this.off("error", this._boundHandleError);
      this._boundHandleError = null;
    }

    // Clean up client_disconnected subscription
    if (this._clientDisconnectedSubscription) {
      try {
        if (
          typeof this._clientDisconnectedSubscription.unsubscribe === "function"
        ) {
          this._clientDisconnectedSubscription.unsubscribe();
        }
      } catch (e) {
        console.debug(`Error unsubscribing client_disconnected: ${e}`);
      }
      this.off("client_disconnected");
      this._clientDisconnectedSubscription = null;
    }

    // Remove the global unhandled rejection handler
    this._removeRejectionHandler();

    // Remove ALL remaining event handlers to prevent memory leaks
    // This clears any custom event listeners registered by users via .on()
    this.off();

    // Clean up background tasks
    try {
      for (const task of this._background_tasks) {
        if (task && typeof task.cancel === "function") {
          try {
            task.cancel();
          } catch (e) {
            console.debug(`Error canceling background task: ${e}`);
          }
        }
      }
      this._background_tasks.clear();
    } catch (e) {
      console.debug(`Error cleaning up background tasks: ${e}`);
    }

    // Clear session sweep interval
    if (this._sessionSweepInterval) {
      clearInterval(this._sessionSweepInterval);
      this._sessionSweepInterval = null;
    }

    // Clean up connection references to prevent circular references
    try {
      this._connection = null;
      this._emit_message = function () {
        console.debug("RPC connection closed, ignoring message");
        return Promise.reject(new Error("Connection is closed"));
      };
    } catch (e) {
      console.debug(`Error during connection cleanup: ${e}`);
    }

    this._fire("disconnected");
  }

  async _handleClientDisconnected(clientId) {
    try {
      console.debug(`Handling disconnection for client: ${clientId}`);

      // Clean up all sessions for the disconnected client
      const sessionsCleaned = this._cleanupSessionsForClient(clientId);

      if (sessionsCleaned > 0) {
        console.debug(
          `Cleaned up ${sessionsCleaned} sessions for disconnected client: ${clientId}`,
        );
      }

      // Fire an event to notify about the client disconnection
      this._fire("remote_client_disconnected", {
        client_id: clientId,
        sessions_cleaned: sessionsCleaned,
      });
    } catch (e) {
      console.error(
        `Error handling client disconnection for ${clientId}: ${e}`,
      );
    }
  }

  _removeFromTargetIdIndex(sessionId) {
    /**
     * Remove a session from the target_id index.
     * Call this before removing a session from _object_store.
     */
    const topKey = sessionId.split(".")[0];
    const session = this._object_store[topKey];
    if (session && typeof session === "object") {
      const targetId = session.target_id;
      if (targetId && targetId in this._targetIdIndex) {
        this._targetIdIndex[targetId].delete(topKey);
        if (this._targetIdIndex[targetId].size === 0) {
          delete this._targetIdIndex[targetId];
        }
      }
    }
  }

  /**
   * Clean up a single session entry: reject promise, clear timer, cancel heartbeat.
   * Centralizes the cleanup logic used by multiple methods.
   * @param {object} session - The session object from _object_store
   * @param {string|null} rejectReason - If provided, reject the session's promise with this reason
   */
  _cleanupSessionEntry(session, rejectReason = null) {
    if (!session || typeof session !== "object") return;
    if (
      rejectReason &&
      session.reject &&
      typeof session.reject === "function"
    ) {
      try {
        session.reject(new Error(rejectReason));
      } catch (e) {
        console.debug(`Error rejecting session: ${e}`);
      }
    }
    if (session.heartbeat_task) {
      try {
        clearInterval(session.heartbeat_task);
      } catch (e) {
        /* ignore */
      }
    }
    if (
      session.timer &&
      session.timer.started &&
      typeof session.timer.clear === "function"
    ) {
      try {
        session.timer.clear();
      } catch (e) {
        /* ignore */
      }
    }
  }

  _cleanupSessionsForClient(clientId) {
    let sessionsCleaned = 0;

    // Use index for O(1) lookup instead of iterating all sessions
    const sessionKeys = this._targetIdIndex[clientId];
    if (!sessionKeys) return 0;

    const reason = `Client disconnected: ${clientId}`;
    for (const sessionKey of sessionKeys) {
      const session = this._object_store[sessionKey];
      if (!session || typeof session !== "object") continue;
      if (session.target_id !== clientId) continue;

      this._cleanupSessionEntry(session, reason);
      delete this._object_store[sessionKey];
      sessionsCleaned++;
      console.debug(`Cleaned up session: ${sessionKey}`);
    }

    delete this._targetIdIndex[clientId];
    return sessionsCleaned;
  }

  _rejectPendingCalls(reason = "Connection lost") {
    /**
     * Reject all pending RPC calls when the connection is lost.
     * Does NOT remove sessions (connection might be re-established).
     */
    try {
      let rejectedCount = 0;
      for (const key of Object.keys(this._object_store)) {
        if (key === "services" || key === "message_cache") continue;
        const value = this._object_store[key];
        if (typeof value === "object" && value !== null) {
          if (value.reject && typeof value.reject === "function") {
            rejectedCount++;
          }
          this._cleanupSessionEntry(value, reason);
        }
      }
      if (rejectedCount > 0) {
        console.warn(
          `Rejected ${rejectedCount} pending RPC call(s) due to: ${reason}`,
        );
      }
    } catch (e) {
      console.error(`Error rejecting pending calls: ${e}`);
    }
  }

  _cleanupOnDisconnect() {
    try {
      console.debug("Cleaning up all sessions due to local RPC disconnection");

      const keysToDelete = [];
      for (const key of Object.keys(this._object_store)) {
        if (key === "services" || key === "message_cache") continue;
        const value = this._object_store[key];
        this._cleanupSessionEntry(value, "RPC connection closed");
        keysToDelete.push(key);
      }

      for (const key of keysToDelete) {
        delete this._object_store[key];
      }

      this._targetIdIndex = {};
    } catch (e) {
      console.error(`Error during cleanup on disconnect: ${e}`);
    }
  }

  async disconnect() {
    // Store connection reference before closing
    const connection = this._connection;
    this.close();

    // Disconnect the underlying connection if it exists
    if (connection) {
      try {
        await connection.disconnect();
      } catch (e) {
        console.debug(`Error disconnecting underlying connection: ${e}`);
      }
    }
  }

  async get_manager_service(config, maxRetries = 20) {
    config = config || {};
    const baseDelay = 500;
    const maxDelay = 10000;
    let lastError = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const retryDelay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);

      if (!this._connection.manager_id) {
        if (attempt < maxRetries - 1) {
          console.warn(
            `Manager ID not set, retrying in ${retryDelay}ms (attempt ${attempt + 1}/${maxRetries})`,
          );
          await new Promise((resolve) => setTimeout(resolve, retryDelay));
          continue;
        } else {
          throw new Error("Manager ID not set after maximum retries");
        }
      }

      try {
        const svc = await this.get_remote_service(
          `*/${this._connection.manager_id}:default`,
          config,
        );
        return svc;
      } catch (e) {
        lastError = e;
        console.warn(
          `Failed to get manager service (attempt ${attempt + 1}/${maxRetries}): ${e.message}`,
        );
        if (attempt < maxRetries - 1) {
          await new Promise((resolve) => setTimeout(resolve, retryDelay));
        }
      }
    }

    throw lastError;
  }

  get_all_local_services() {
    return this._services;
  }
  get_local_service(service_id, context) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(service_id);
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(context, "Context is required");

    const [ws, client_id] = context["to"].split("/");
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      client_id === this._client_id,
      "Services can only be accessed locally",
    );

    const service = this._services[service_id];
    if (!service) {
      throw new Error("Service not found: " + service_id);
    }

    // Note: Do NOT mutate service.config.workspace here!
    // Doing so would corrupt the stored service config when called from
    // a different workspace (e.g., "public"), causing reconnection to fail
    // because _extract_service_info would use the wrong workspace value.

    // allow access for the same workspace
    if (
      service.config.visibility == "public" ||
      service.config.visibility == "unlisted"
    ) {
      return service;
    }

    // allow access for the same workspace
    if (context["ws"] === ws) {
      return service;
    }

    // Check if user is from an authorized workspace
    const authorized_workspaces = service.config.authorized_workspaces;
    if (
      authorized_workspaces &&
      authorized_workspaces.includes(context["ws"])
    ) {
      return service;
    }

    throw new Error(
      `Permission denied for getting protected service: ${service_id}, workspace mismatch: ${ws} != ${context["ws"]}`,
    );
  }
  async get_remote_service(service_uri, config) {
    let { timeout, case_conversion, kwargs_expansion, encryption_public_key } =
      config || {};
    timeout = timeout === undefined ? this._method_timeout : timeout;
    if (!service_uri && this._connection.manager_id) {
      service_uri = "*/" + this._connection.manager_id;
    } else if (!service_uri.includes(":")) {
      service_uri = this._client_id + ":" + service_uri;
    }
    const provider = service_uri.split(":")[0];
    let service_id = service_uri.split(":")[1];
    if (service_id.includes("@")) {
      service_id = service_id.split("@")[0];
      const app_id = service_uri.split("@")[1];
      if (this._app_id && this._app_id !== "*")
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
          app_id === this._app_id,
          `Invalid app id: ${app_id} != ${this._app_id}`,
        );
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(provider, `Invalid service uri: ${service_uri}`);

    try {
      const method = this._generate_remote_method({
        _rserver: this._server_base_url,
        _rtarget: provider,
        _rmethod: "services.built-in.get_service",
        _rpromise: true,
        _rdoc: "Get a remote service",
      });
      let svc = await (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.waitFor)(
        method(service_id),
        timeout,
        "Timeout Error: Failed to get remote service: " + service_uri,
      );
      svc.id = `${provider}:${service_id}`;
      if (encryption_public_key) {
        const encPubBytes = (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.publicKeyFromHex)(encryption_public_key);
        _applyEncryptionKeyToService(svc, encPubBytes);
      }
      if (kwargs_expansion) {
        svc = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.expandKwargs)(svc);
      }
      if (case_conversion)
        return Object.assign(
          new RemoteService(),
          (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.convertCase)(svc, case_conversion),
        );
      else return Object.assign(new RemoteService(), svc);
    } catch (e) {
      console.warn("Failed to get remote service: " + service_uri, e);
      throw e;
    }
  }
  _annotate_service_methods(
    aObject,
    object_id,
    require_context,
    run_in_executor,
    visibility,
    authorized_workspaces,
    trusted_keys,
  ) {
    if (typeof aObject === "function") {
      // mark the method as a remote method that requires context
      let method_name = object_id.split(".")[1];
      this._method_annotations.set(aObject, {
        require_context: Array.isArray(require_context)
          ? require_context.includes(method_name)
          : !!require_context,
        run_in_executor: run_in_executor,
        method_id: "services." + object_id,
        visibility: visibility,
        authorized_workspaces: authorized_workspaces,
        trusted_keys: trusted_keys,
      });
    } else if (aObject instanceof Array || aObject instanceof Object) {
      for (let key of Object.keys(aObject)) {
        let val = aObject[key];
        if (typeof val === "function" && val.__rpc_object__) {
          let client_id = val.__rpc_object__._rtarget;
          if (client_id.includes("/")) {
            client_id = client_id.split("/")[1];
          }
          if (this._client_id === client_id) {
            if (aObject instanceof Array) {
              aObject = aObject.slice();
            }
            // recover local method
            aObject[key] = indexObject(
              this._object_store,
              val.__rpc_object__._rmethod,
            );
            val = aObject[key]; // make sure it's annotated later
          } else {
            throw new Error(
              `Local method not found: ${val.__rpc_object__._rmethod}, client id mismatch ${this._client_id} != ${client_id}`,
            );
          }
        }
        this._annotate_service_methods(
          val,
          object_id + "." + key,
          require_context,
          run_in_executor,
          visibility,
          authorized_workspaces,
          trusted_keys,
        );
      }
    }
  }
  add_service(api, overwrite) {
    if (!api || Array.isArray(api)) throw new Error("Invalid service object");
    if (api.constructor === Object) {
      api = Object.assign({}, api);
    } else {
      const normApi = {};
      const props = Object.getOwnPropertyNames(api).concat(
        Object.getOwnPropertyNames(Object.getPrototypeOf(api)),
      );
      for (let k of props) {
        if (k !== "constructor") {
          if (typeof api[k] === "function") normApi[k] = api[k].bind(api);
          else normApi[k] = api[k];
        }
      }
      // For class instance, we need set a default id
      api.id = api.id || "default";
      api = normApi;
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      api.id && typeof api.id === "string",
      `Service id not found: ${api}`,
    );
    if (!api.name) {
      api.name = api.id;
    }
    if (!api.config) {
      api.config = {};
    }
    if (!api.type) {
      api.type = "generic";
    }
    // require_context only applies to the top-level functions
    let require_context = false,
      run_in_executor = false;
    if (api.config.require_context)
      require_context = api.config.require_context;
    if (api.config.run_in_executor) run_in_executor = true;
    const visibility = api.config.visibility || "protected";
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(["protected", "public", "unlisted"].includes(visibility));

    // Validate authorized_workspaces
    const authorized_workspaces = api.config.authorized_workspaces;
    if (authorized_workspaces !== undefined) {
      if (visibility !== "protected") {
        throw new Error(
          `authorized_workspaces can only be set when visibility is 'protected', got visibility='${visibility}'`,
        );
      }
      if (!Array.isArray(authorized_workspaces)) {
        throw new Error(
          "authorized_workspaces must be an array of workspace ids",
        );
      }
      for (const ws_id of authorized_workspaces) {
        if (typeof ws_id !== "string") {
          throw new Error(
            `Each workspace id in authorized_workspaces must be a string, got ${typeof ws_id}`,
          );
        }
      }
    }
    const trustedKeysHex = api.config.trusted_keys;
    let trusted_keys = null;
    if (trustedKeysHex && trustedKeysHex.length > 0) {
      if (!Array.isArray(trustedKeysHex)) {
        throw new Error(
          "trusted_keys must be an array of hex-encoded public keys",
        );
      }
      trusted_keys = new Set();
      for (const keyHex of trustedKeysHex) {
        if (typeof keyHex !== "string" || keyHex.length !== 64) {
          throw new Error(
            `Each trusted key must be a 64-char hex string, got: ${keyHex}`,
          );
        }
        trusted_keys.add(keyHex);
      }
    }
    this._annotate_service_methods(
      api,
      api["id"],
      require_context,
      run_in_executor,
      visibility,
      authorized_workspaces,
      trusted_keys,
    );

    if (this._services[api.id]) {
      if (overwrite) {
        delete this._services[api.id];
      } else {
        throw new Error(
          `Service already exists: ${api.id}, please specify a different id (not ${api.id}) or overwrite=true`,
        );
      }
    }
    this._services[api.id] = api;
    return api;
  }

  _extract_service_info(service) {
    const config = service.config || {};
    config.workspace =
      config.workspace || this._local_workspace || this._connection.workspace;
    if (!config.workspace) {
      throw new Error(
        "Workspace is not set. Please ensure the connection has a workspace or set local_workspace.",
      );
    }
    const skipContext = config.require_context;
    const excludeKeys = [
      "id",
      "config",
      "name",
      "description",
      "type",
      "docs",
      "app_id",
      "service_schema",
    ];
    const filteredService = {};
    for (const key of Object.keys(service)) {
      if (!excludeKeys.includes(key)) {
        filteredService[key] = service[key];
      }
    }
    const serviceSchema = _get_schema(filteredService, null, skipContext);
    const serviceInfo = {
      config: config,
      id: `${config.workspace}/${this._client_id}:${service["id"]}`,
      name: service.name || service["id"],
      description: service.description || "",
      type: service.type || "generic",
      docs: service.docs || null,
      app_id: this._app_id,
      service_schema: serviceSchema,
    };
    return serviceInfo;
  }

  async get_service_schema(service) {
    const skipContext = service.config.require_context;
    return _get_schema(service, null, skipContext);
  }

  async register_service(api, config) {
    let { check_type, notify, overwrite } = config || {};
    notify = notify === undefined ? true : notify;
    let manager;
    if (check_type && api.type) {
      try {
        manager = await this.get_manager_service({
          timeout: 10,
          case_conversion: "camel",
        });
        const type_info = await manager.get_service_type(api.type);
        api = _annotate_service(api, type_info);
      } catch (e) {
        throw new Error(`Failed to get service type ${api.type}, error: ${e}`);
      }
    }

    const service = this.add_service(api, overwrite);
    const serviceInfo = this._extract_service_info(service);
    if (notify) {
      try {
        manager =
          manager ||
          (await this.get_manager_service({
            timeout: 10,
            case_conversion: "camel",
          }));
        await manager.registerService(serviceInfo);
      } catch (e) {
        throw new Error(`Failed to notify workspace manager: ${e}`);
      }
    }
    return serviceInfo;
  }

  async unregister_service(service, notify) {
    notify = notify === undefined ? true : notify;
    let service_id;
    if (typeof service === "string") {
      service_id = service;
    } else {
      service_id = service.id;
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      service_id && typeof service_id === "string",
      `Invalid service id: ${service_id}`,
    );
    if (service_id.includes(":")) {
      service_id = service_id.split(":")[1];
    }
    if (service_id.includes("@")) {
      service_id = service_id.split("@")[0];
    }
    if (!this._services[service_id]) {
      throw new Error(`Service not found: ${service_id}`);
    }
    // Auto-detect _rintf services (local-only, never registered with server)
    if (service_id.startsWith("_rintf_")) {
      notify = false;
    }
    if (notify) {
      const manager = await this.get_manager_service({
        timeout: 10,
        case_conversion: "camel",
      });
      await manager.unregisterService(service_id);
    }
    delete this._services[service_id];
  }

  _ndarray(typedArray, shape, dtype) {
    const _dtype = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.typedArrayToDtype)(typedArray);
    if (dtype && dtype !== _dtype) {
      throw (
        "dtype doesn't match the type of the array: " + _dtype + " != " + dtype
      );
    }
    shape = shape || [typedArray.length];
    return {
      _rtype: "ndarray",
      _rvalue: typedArray.buffer,
      _rshape: shape,
      _rdtype: _dtype,
    };
  }

  _encode_callback(
    name,
    callback,
    session_id,
    clear_after_called,
    timer,
    local_workspace,
    description,
  ) {
    let method_id = `${session_id}.${name}`;
    let encoded = {
      _rtype: "method",
      _rtarget: local_workspace
        ? `${local_workspace}/${this._client_id}`
        : this._client_id,
      _rmethod: method_id,
      _rpromise: false,
    };

    const self = this;
    let wrapped_callback = function () {
      try {
        callback.apply(null, Array.prototype.slice.call(arguments));
      } catch (error) {
        console.error(
          `Error in callback(${method_id}, ${description}): ${error}`,
        );
      } finally {
        // Clear the timer first if it exists
        if (timer && timer.started) {
          timer.clear();
        }

        // Clean up the entire session when resolve/reject is called
        if (clear_after_called && self._object_store[session_id]) {
          if (name === "resolve" || name === "reject") {
            self._removeFromTargetIdIndex(session_id);
            delete self._object_store[session_id];
          } else {
            self._cleanup_session_if_needed(session_id, name);
          }
        }
      }
    };
    wrapped_callback.__name__ = `callback(${method_id})`;
    return [encoded, wrapped_callback];
  }

  _cleanup_session_if_needed(session_id, callback_name) {
    /**
     * Clean session management - all logic in one place.
     */
    if (!session_id) {
      console.debug("Cannot cleanup session: session_id is empty");
      return;
    }

    try {
      const store = this._get_session_store(session_id, false);
      if (!store) {
        console.debug(`Session ${session_id} not found for cleanup`);
        return;
      }

      let should_cleanup = false;

      // Promise sessions: let the promise manager decide cleanup
      if (store._promise_manager) {
        try {
          const promise_manager = store._promise_manager;
          if (
            promise_manager.should_cleanup_on_callback &&
            promise_manager.should_cleanup_on_callback(callback_name)
          ) {
            if (promise_manager.settle) {
              promise_manager.settle();
            }
            should_cleanup = true;
            console.debug(
              `Promise session ${session_id} settled and marked for cleanup`,
            );
          }
        } catch (e) {
          console.warn(
            `Error in promise manager cleanup for session ${session_id}:`,
            e,
          );
        }
      } else {
        // Regular sessions: only cleanup temporary method call sessions
        // Don't cleanup service registration sessions or persistent sessions
        // Only cleanup sessions that are clearly temporary promises for method calls
        if (
          (callback_name === "resolve" || callback_name === "reject") &&
          store._callbacks &&
          Object.keys(store._callbacks).includes(callback_name)
        ) {
          should_cleanup = true;
          console.debug(
            `Regular session ${session_id} marked for cleanup after ${callback_name}`,
          );
        }
      }

      if (should_cleanup) {
        this._cleanup_session_completely(session_id);
      }
    } catch (error) {
      console.warn(`Error during session cleanup for ${session_id}:`, error);
    }
  }

  _cleanup_session_completely(session_id) {
    /**
     * Complete session cleanup with resource management.
     */
    try {
      // Clean up target_id index before deleting the session
      this._removeFromTargetIdIndex(session_id);

      const store = this._get_session_store(session_id, false);
      if (!store) {
        console.debug(`Session ${session_id} already cleaned up`);
        return;
      }

      // Clean up resources before removing session
      if (
        store.timer &&
        store.timer.started &&
        typeof store.timer.clear === "function"
      ) {
        try {
          store.timer.clear();
        } catch (error) {
          console.warn(
            `Error clearing timer for session ${session_id}:`,
            error,
          );
        }
      }

      if (
        store.heartbeat_task &&
        typeof store.heartbeat_task.cancel === "function"
      ) {
        try {
          store.heartbeat_task.cancel();
        } catch (error) {
          console.warn(
            `Error canceling heartbeat for session ${session_id}:`,
            error,
          );
        }
      }

      // Navigate and clean session path
      const levels = session_id.split(".");
      let current_store = this._object_store;

      // Navigate to parent of target level
      for (let i = 0; i < levels.length - 1; i++) {
        const level = levels[i];
        if (!current_store[level]) {
          console.debug(
            `Session path ${session_id} not found at level ${level}`,
          );
          return;
        }
        current_store = current_store[level];
      }

      // Delete the final level
      const final_key = levels[levels.length - 1];
      if (current_store[final_key]) {
        delete current_store[final_key];
        console.debug(`Cleaned up session ${session_id}`);

        // Clean up empty parent containers
        this._cleanup_empty_containers(levels.slice(0, -1));
      }
    } catch (error) {
      console.warn(
        `Error in complete session cleanup for ${session_id}:`,
        error,
      );
    }
  }

  _cleanup_empty_containers(path_levels) {
    /**
     * Clean up empty parent containers to prevent memory leaks.
     */
    try {
      // Work backwards from the deepest level
      for (let depth = path_levels.length - 1; depth >= 0; depth--) {
        let current_store = this._object_store;

        // Navigate to parent of current depth
        for (let i = 0; i < depth; i++) {
          current_store = current_store[path_levels[i]];
          if (!current_store) return; // Path doesn't exist
        }

        // Check if container at current depth is empty
        const container_key = path_levels[depth];
        const container = current_store[container_key];

        if (
          container &&
          typeof container === "object" &&
          Object.keys(container).length === 0
        ) {
          delete current_store[container_key];
          console.debug(
            `Cleaned up empty container at depth ${depth}: ${path_levels.slice(0, depth + 1).join(".")}`,
          );
        } else {
          // Container is not empty, stop cleanup
          break;
        }
      }
    } catch (error) {
      console.warn("Error cleaning up empty containers:", error);
    }
  }

  get_session_stats() {
    /**
     * Get detailed session statistics.
     */
    const stats = {
      total_sessions: 0,
      promise_sessions: 0,
      regular_sessions: 0,
      sessions_with_timers: 0,
      sessions_with_heartbeat: 0,
      system_stores: {},
      session_ids: [],
      memory_usage: 0,
    };

    if (!this._object_store) {
      return stats;
    }

    for (const key in this._object_store) {
      const value = this._object_store[key];

      if (["services", "message_cache"].includes(key)) {
        // System stores - don't count these as sessions
        stats.system_stores[key] = {
          size:
            typeof value === "object" && value ? Object.keys(value).length : 0,
        };
        continue;
      }

      // Count all non-system non-empty objects as sessions
      if (value && typeof value === "object") {
        const sessionKeys = Object.keys(value);

        // Only skip completely empty objects
        if (sessionKeys.length > 0) {
          stats.total_sessions++;
          stats.session_ids.push(key);

          if (value._promise_manager) {
            stats.promise_sessions++;
          } else {
            stats.regular_sessions++;
          }

          if (value._timer || value.timer) stats.sessions_with_timers++;
          if (value._heartbeat || value.heartbeat)
            stats.sessions_with_heartbeat++;

          // Estimate memory usage
          stats.memory_usage += JSON.stringify(value).length;
        }
      }
    }

    return stats;
  }

  _force_cleanup_all_sessions() {
    /**
     * Force cleanup all sessions (for testing purposes).
     */
    if (!this._object_store) {
      console.debug("Force cleaning up 0 sessions");
      return;
    }

    let cleaned_count = 0;
    const keys_to_delete = [];

    for (const key in this._object_store) {
      // Don't delete system stores
      if (!["services", "message_cache"].includes(key)) {
        const value = this._object_store[key];
        if (
          value &&
          typeof value === "object" &&
          Object.keys(value).length > 0
        ) {
          keys_to_delete.push(key);
          cleaned_count++;
        }
      }
    }

    // Delete the sessions
    for (const key of keys_to_delete) {
      delete this._object_store[key];
    }

    // Clear the target_id index since all sessions are removed
    this._targetIdIndex = {};

    console.debug(`Force cleaning up ${cleaned_count} sessions`);
  }

  _sweepStaleSessions() {
    const now = Date.now();
    let swept = 0;
    for (const key of Object.keys(this._object_store)) {
      if (key === "services" || key === "message_cache") continue;
      const session = this._object_store[key];
      // Use last-activity time if available, fall back to creation time
      const lastActivity =
        session && (session._last_activity_at || session._created_at);
      if (
        session &&
        typeof session === "object" &&
        lastActivity &&
        now - lastActivity > this._sessionMaxAge
      ) {
        // Only sweep sessions that have no timer (active timers mean they are in use)
        // and no active promise callbacks (resolve/reject mean the session is awaiting a response)
        if (!session.timer || !session.timer.started) {
          if (
            typeof session.resolve === "function" ||
            typeof session.reject === "function"
          ) {
            // Session still has active promise callbacks, skip it
            continue;
          }
          this._removeFromTargetIdIndex(key);
          if (session.heartbeat_task) clearInterval(session.heartbeat_task);
          delete this._object_store[key];
          swept++;
        }
      }
    }
    if (swept > 0) {
      console.debug(`Swept ${swept} stale session(s)`);
    }
  }

  // Clean helper to identify promise method calls by session type
  _is_promise_method_call(method_path) {
    const session_id = method_path.split(".")[0];
    const session = this._get_session_store(session_id, false);
    return session && session._promise_manager;
  }

  // Simplified Promise Manager - enhanced version
  _create_promise_manager() {
    /**
     * Create a promise manager to track promise state and decide cleanup.
     */
    return {
      should_cleanup_on_callback: (callback_name) => {
        return ["resolve", "reject"].includes(callback_name);
      },
      settle: () => {
        // Promise is settled (resolved or rejected)
        console.debug("Promise settled");
      },
    };
  }

  async _encode_promise(
    resolve,
    reject,
    session_id,
    clear_after_called,
    timer,
    local_workspace,
    description,
  ) {
    let store = this._get_session_store(session_id, true);
    if (!store) {
      console.warn(
        `Failed to create session store ${session_id}, session management may be impaired`,
      );
      store = {};
    }

    // Clean promise lifecycle management - TYPE-BASED, not string-based
    store._promise_manager = this._create_promise_manager();

    let encoded = {};

    if (timer && reject && this._method_timeout) {
      [encoded.heartbeat, store.heartbeat] = this._encode_callback(
        "heartbeat",
        timer.reset.bind(timer),
        session_id,
        false,
        null,
        local_workspace,
      );
      store.timer = timer;
      encoded.interval = this._method_timeout / 2;
    } else {
      timer = null;
    }

    [encoded.resolve, store.resolve] = this._encode_callback(
      "resolve",
      resolve,
      session_id,
      clear_after_called,
      timer,
      local_workspace,
      `resolve (${description})`,
    );
    [encoded.reject, store.reject] = this._encode_callback(
      "reject",
      reject,
      session_id,
      clear_after_called,
      timer,
      local_workspace,
      `reject (${description})`,
    );
    return encoded;
  }

  async _send_chunks(data, target_id, session_id) {
    // 1) Get the remote service
    const remote_services = await this.get_remote_service(
      `${target_id}:built-in`,
    );
    if (!remote_services.message_cache) {
      throw new Error(
        "Remote client does not support message caching for large messages.",
      );
    }

    const message_cache = remote_services.message_cache;
    const message_id = session_id || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
    const total_size = data.length;
    const start_time = Date.now(); // measure time
    const chunk_num = Math.ceil(total_size / this._long_message_chunk_size);
    if (remote_services.config.api_version >= 3) {
      await message_cache.create(message_id, !!session_id);
      const semaphore = new _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.Semaphore(CONCURRENCY_LIMIT);

      const tasks = [];
      for (let idx = 0; idx < chunk_num; idx++) {
        const startByte = idx * this._long_message_chunk_size;
        const chunk = data.slice(
          startByte,
          startByte + this._long_message_chunk_size,
        );

        const taskFn = async () => {
          await message_cache.set(message_id, idx, chunk, !!session_id);
          // console.debug(
          //   `Sending chunk ${idx + 1}/${chunk_num} (total=${total_size} bytes)`,
          // );
        };

        // Push into an array, each one runs under the semaphore
        tasks.push(semaphore.run(taskFn));
      }

      // Wait for all chunk uploads to finish
      try {
        await Promise.all(tasks);
      } catch (error) {
        // If any chunk fails, clean up the message cache
        try {
          await message_cache.remove(message_id);
        } catch (cleanupError) {
          console.error(
            `Failed to clean up message cache after error: ${cleanupError}`,
          );
        }
        throw error;
      }
    } else {
      // 3) Legacy version (sequential appends):
      await message_cache.create(message_id, !!session_id);
      for (let idx = 0; idx < chunk_num; idx++) {
        const startByte = idx * this._long_message_chunk_size;
        const chunk = data.slice(
          startByte,
          startByte + this._long_message_chunk_size,
        );
        await message_cache.append(message_id, chunk, !!session_id);
        // console.debug(
        //   `Sending chunk ${idx + 1}/${chunk_num} (total=${total_size} bytes)`,
        // );
      }
    }
    await message_cache.process(message_id, !!session_id);
    const durationSec = ((Date.now() - start_time) / 1000).toFixed(2);
    // console.debug(`All chunks (${total_size} bytes) sent in ${durationSec} s`);
  }

  emit(main_message, extra_data) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
      typeof main_message === "object" && main_message.type,
      "Invalid message, must be an object with a `type` fields.",
    );
    if (!main_message.to) {
      this._fire(main_message.type, main_message);
      return;
    }

    let message_package = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.encode)(main_message);
    if (extra_data) {
      const extra = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.encode)(extra_data);
      const combined = new Uint8Array(message_package.length + extra.length);
      combined.set(message_package);
      combined.set(extra, message_package.length);
      message_package = combined;
    }
    const total_size = message_package.length;
    if (total_size > this._long_message_chunk_size + 1024) {
      console.warn(`Sending large message (size=${total_size})`);
    }
    return this._emit_message(message_package);
  }

  _generate_remote_method(
    encoded_method,
    remote_parent,
    local_parent,
    remote_workspace,
    local_workspace,
  ) {
    let target_id = encoded_method._rtarget;
    if (remote_workspace && !target_id.includes("/")) {
      // Don't modify target_id if it starts with */ (workspace manager service)
      if (!target_id.startsWith("*/")) {
        if (remote_workspace !== target_id) {
          target_id = remote_workspace + "/" + target_id;
        }
        // Fix the target id to be an absolute id
        encoded_method._rtarget = target_id;
      }
    }
    let method_id = encoded_method._rmethod;
    let with_promise = encoded_method._rpromise || false;
    const description = `method: ${method_id}, docs: ${encoded_method._rdoc}`;
    const self = this;

    function remote_method() {
      return new Promise(async (resolve, reject) => {
        try {
          let local_session_id = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
          if (local_parent) {
            // Store the children session under the parent
            local_session_id = local_parent + "." + local_session_id;
          }
          let store = self._get_session_store(local_session_id, true);
          if (!store) {
            reject(
              new Error(
                `Runtime Error: Failed to get session store ${local_session_id} (context: ${description})`,
              ),
            );
            return;
          }
          store["target_id"] = target_id;
          // Update target_id index for fast session cleanup
          const topKey = local_session_id.split(".")[0];
          if (!(target_id in self._targetIdIndex)) {
            self._targetIdIndex[target_id] = new Set();
          }
          self._targetIdIndex[target_id].add(topKey);
          const args = await self._encode(
            Array.prototype.slice.call(arguments),
            local_session_id,
            local_workspace,
          );
          const argLength = args.length;
          // if the last argument is an object, mark it as kwargs
          const withKwargs =
            argLength > 0 &&
            typeof args[argLength - 1] === "object" &&
            args[argLength - 1] !== null &&
            args[argLength - 1]._rkwargs;
          if (withKwargs) delete args[argLength - 1]._rkwargs;

          let from_client;
          if (!self._local_workspace) {
            from_client = self._client_id;
          } else {
            from_client = self._local_workspace + "/" + self._client_id;
          }

          let main_message = {
            type: "method",
            from: from_client,
            to: target_id,
            method: method_id,
          };
          let extra_data = {};
          if (args) {
            extra_data["args"] = args;
          }
          if (withKwargs) {
            extra_data["with_kwargs"] = withKwargs;
          }

          // console.log(
          //   `Calling remote method ${target_id}:${method_id}, session: ${local_session_id}`
          // );
          if (remote_parent) {
            // Set the parent session
            // Note: It's a session id for the remote, not the current client
            main_message["parent"] = remote_parent;
          }

          let timer = null;
          if (with_promise) {
            // Only pass the current session id to the remote
            // if we want to received the result
            // I.e. the session id won't be passed for promises themselves
            main_message["session"] = local_session_id;
            let method_name = `${target_id}:${method_id}`;

            // Create a timer that gets reset by heartbeat
            // Methods can run indefinitely as long as heartbeat keeps resetting the timer
            // IMPORTANT: When timeout occurs, we must clean up the session to prevent memory leaks
            const timeoutCallback = function (error_msg) {
              // First reject the promise - wrap in Error for proper stack traces
              reject(new Error(error_msg));
              // Then clean up the entire session to stop all callbacks
              if (self._object_store[local_session_id]) {
                // Clean up target_id index before deleting the session
                self._removeFromTargetIdIndex(local_session_id);
                delete self._object_store[local_session_id];
                console.debug(
                  `Cleaned up session ${local_session_id} after timeout`,
                );
              }
            };

            timer = new Timer(
              self._method_timeout,
              timeoutCallback,
              [
                `Method call timed out: ${method_name}, context: ${description}`,
              ],
              method_name,
            );
            let clear_after_called = true;

            const promiseData = await self._encode_promise(
              resolve,
              reject,
              local_session_id,
              clear_after_called,
              timer,
              local_workspace,
              description,
            );

            if (with_promise === true) {
              extra_data["promise"] = promiseData;
            } else if (with_promise === "*") {
              extra_data["promise"] = "*";
              extra_data["t"] = self._method_timeout / 2;
            } else {
              throw new Error(`Unsupported promise type: ${with_promise}`);
            }
          }
          // E2E encrypt extra_data if both sides support encryption
          if (self._encryption_ready) {
            await self._encryption_ready;
          }
          if (
            extra_data &&
            self._encryption_enabled &&
            encoded_method._renc_pub
          ) {
            const plaintext = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.encode)(extra_data);
            const { nonce, ciphertext } = await (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.encryptPayload)(
              self._encryption_private_key,
              new Uint8Array(encoded_method._renc_pub),
              plaintext,
            );
            extra_data = {
              _enc: {
                v: 2,
                pub: self._encryption_public_key,
                nonce: nonce,
              },
              data: ciphertext,
            };
          }

          // The message consists of two segments, the main message and extra data
          let message_package = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.encode)(main_message);
          if (extra_data) {
            const extra = (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_4__.encode)(extra_data);
            const combined = new Uint8Array(
              message_package.length + extra.length,
            );
            combined.set(message_package);
            combined.set(extra, message_package.length);
            message_package = combined;
          }
          const total_size = message_package.length;
          if (
            total_size <= self._long_message_chunk_size + 1024 ||
            remote_method.__no_chunk__
          ) {
            self
              ._emit_message(message_package)
              .then(function () {
                if (timer) {
                  // Start the timer after message is sent successfully
                  timer.start();
                }
                if (!with_promise) {
                  // Fire-and-forget: resolve immediately after message is sent.
                  // Without this, the promise never resolves because no response
                  // is expected. This is critical for heartbeat callbacks which
                  // use _rpromise=false and are awaited in a loop.
                  resolve(null);
                }
              })
              .catch(function (err) {
                const error_msg = `Failed to send the request when calling method (${target_id}:${method_id}), error: ${err}`;
                if (reject) {
                  reject(new Error(error_msg));
                } else {
                  // No reject callback available, log the error to prevent unhandled promise rejections
                  console.warn("Unhandled RPC method call error:", error_msg);
                }
                if (timer && timer.started) {
                  timer.clear();
                }
              });
          } else {
            // send chunk by chunk
            self
              ._send_chunks(message_package, target_id, remote_parent)
              .then(function () {
                if (timer) {
                  // Start the timer after message is sent successfully
                  timer.start();
                }
                if (!with_promise) {
                  // Fire-and-forget: resolve immediately after message is sent
                  resolve(null);
                }
              })
              .catch(function (err) {
                const error_msg = `Failed to send the request when calling method (${target_id}:${method_id}), error: ${err}`;
                if (reject) {
                  reject(new Error(error_msg));
                } else {
                  // No reject callback available, log the error to prevent unhandled promise rejections
                  console.warn("Unhandled RPC method call error:", error_msg);
                }
                if (timer && timer.started) {
                  timer.clear();
                }
              });
          }
        } catch (err) {
          reject(err);
        }
      });
    }

    // Generate debugging information for the method
    remote_method.__rpc_object__ = encoded_method;
    const parts = method_id.split(".");

    remote_method.__name__ = encoded_method._rname || parts[parts.length - 1];
    if (remote_method.__name__.includes("#")) {
      remote_method.__name__ = remote_method.__name__.split("#")[1];
    }
    remote_method.__doc__ =
      encoded_method._rdoc || `Remote method: ${method_id}`;
    remote_method.__schema__ = encoded_method._rschema;
    // Prevent circular chunk sending
    remote_method.__no_chunk__ =
      encoded_method._rmethod === "services.built-in.message_cache.append";
    return remote_method;
  }

  get_client_info() {
    const services = [];
    for (let service of Object.values(this._services)) {
      services.push(this._extract_service_info(service));
    }

    return {
      id: this._client_id,
      services: services,
    };
  }

  async _handle_method(data) {
    let resolve = null;
    let reject = null;
    let heartbeat_task = null;
    try {
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(data.method && data.ctx && data.from);
      const method_name = data.from + ":" + data.method;
      const remote_workspace = data.from.split("/")[0];
      const remote_client_id = data.from.split("/")[1];
      // Make sure the target id is an absolute id
      data["to"] = data["to"].includes("/")
        ? data["to"]
        : remote_workspace + "/" + data["to"];
      data["ctx"]["to"] = data["to"];
      let local_workspace;
      if (!this._local_workspace) {
        local_workspace = data["to"].split("/")[0];
      } else {
        if (this._local_workspace && this._local_workspace !== "*") {
          (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
            data["to"].split("/")[0] === this._local_workspace,
            "Workspace mismatch: " +
              data["to"].split("/")[0] +
              " != " +
              this._local_workspace,
          );
        }
        local_workspace = this._local_workspace;
      }
      const local_parent = data.parent;

      // E2E Decryption: if extra_data was encrypted, decrypt it now
      let callerEncryptionPub = null;
      if (data._enc && this._encryption_enabled) {
        const encInfo = data._enc;
        if (encInfo && typeof encInfo === "object" && encInfo.v === 2) {
          callerEncryptionPub = new Uint8Array(encInfo.pub);
          const nonce = new Uint8Array(encInfo.nonce);
          const ciphertext = new Uint8Array(data.data);
          let plaintext;
          try {
            plaintext = await (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.decryptPayload)(
              this._encryption_private_key,
              callerEncryptionPub,
              nonce,
              ciphertext,
            );
          } catch (e) {
            throw new Error(
              `Decryption failed for method ${method_name}. Invalid key or tampered data.`,
            );
          }
          // Unpack original extra_data and merge back
          const decoded = [];
          for (const item of (0,_msgpack_msgpack__WEBPACK_IMPORTED_MODULE_3__.decodeMulti)(plaintext)) {
            decoded.push(item);
          }
          const originalExtra = decoded[0];
          delete data._enc;
          delete data.data;
          Object.assign(data, originalExtra);
          // Store caller's encryption pub in context
          data.ctx.encryption = true;
          data.ctx.caller_public_key = (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.publicKeyToHex)(callerEncryptionPub);
        }
      }

      if (data.promise) {
        // Decode the promise with the remote session id
        // Such that the session id will be passed to the remote as a parent session id
        const promise = await this._decode(
          data.promise === "*"
            ? this._expand_promise(data, callerEncryptionPub)
            : data.promise,
          data.session,
          local_parent,
          remote_workspace,
          local_workspace,
        );
        resolve = promise.resolve;
        reject = promise.reject;
        if (promise.heartbeat && promise.interval) {
          async function heartbeat() {
            try {
              // console.debug("Reset heartbeat timer: " + data.method);
              await promise.heartbeat();
            } catch (err) {
              console.error(err);
            }
          }
          heartbeat_task = setInterval(heartbeat, promise.interval * 1000);
          // Store the heartbeat task in the session store for cleanup
          if (data.session) {
            const session_store = this._get_session_store(data.session, false);
            if (session_store) {
              session_store.heartbeat_task = heartbeat_task;
            }
          }
        }
      }

      let method;

      try {
        method = indexObject(this._object_store, data["method"]);
        // Update last-activity time for session GC
        const methodParts = data["method"].split(".");
        if (methodParts.length > 1) {
          const topKey = methodParts[0];
          if (topKey !== "services" && topKey !== "message_cache") {
            // Skip system stores — they are not GC-managed sessions
            const topSession = this._object_store[topKey];
            if (topSession && typeof topSession === "object") {
              topSession._last_activity_at = Date.now();
            }
          }
        }
      } catch (e) {
        // Clean promise method detection - TYPE-BASED, not string-based
        if (this._is_promise_method_call(data["method"])) {
          console.debug(
            `Promise method ${data["method"]} not available (detected by session type), ignoring: ${method_name}`,
          );
          return;
        }

        // Check if this is a session-based method call that might have expired
        const method_parts = data["method"].split(".");
        if (method_parts.length > 1) {
          const session_id = method_parts[0];
          // Check if the session exists but the specific method doesn't
          if (session_id in this._object_store) {
            console.debug(
              `Session ${session_id} exists but method ${data["method"]} not found, likely expired callback: ${method_name}`,
            );
            // For expired callbacks, don't throw an exception, just log and return
            if (typeof reject === "function") {
              reject(new Error(`Method expired or not found: ${method_name}`));
            }
            return;
          } else {
            console.debug(
              `Session ${session_id} not found for method ${data["method"]}, likely cleaned up: ${method_name}`,
            );
            // For cleaned up sessions, just log and return without throwing
            if (typeof reject === "function") {
              reject(new Error(`Session not found: ${method_name}`));
            }
            return;
          }
        }

        console.debug(
          `Failed to find method ${method_name} at ${this._client_id}`,
        );
        const error = new Error(
          `Method not found: ${method_name} at ${this._client_id}`,
        );
        if (typeof reject === "function") {
          reject(error);
        } else {
          // Log the error instead of throwing to prevent unhandled exceptions
          console.warn(
            "Method not found and no reject callback:",
            error.message,
          );
        }
        return;
      }

      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        method && typeof method === "function",
        "Invalid method: " + method_name,
      );

      // Check permission
      if (this._method_annotations.has(method)) {
        // For services, it should not be protected
        if (this._method_annotations.get(method).visibility === "protected") {
          // Allow access from same workspace
          if (local_workspace === remote_workspace) {
            // Access granted
          }
          // Check if remote workspace is in authorized_workspaces list
          else if (
            this._method_annotations.get(method).authorized_workspaces &&
            this._method_annotations
              .get(method)
              .authorized_workspaces.includes(remote_workspace)
          ) {
            // Access granted
          }
          // Allow manager access
          else if (
            remote_workspace === "*" &&
            remote_client_id === this._connection.manager_id
          ) {
            // Access granted
          } else {
            throw new Error(
              "Permission denied for invoking protected method " +
                method_name +
                ", workspace mismatch: " +
                local_workspace +
                " != " +
                remote_workspace,
            );
          }
        }
      } else {
        // For sessions, the target_id should match exactly
        let session_target_id =
          this._object_store[data.method.split(".")[0]].target_id;
        if (
          local_workspace === remote_workspace &&
          session_target_id &&
          session_target_id.indexOf("/") === -1
        ) {
          session_target_id = local_workspace + "/" + session_target_id;
        }
        if (session_target_id !== data.from) {
          throw new Error(
            "Access denied for method call (" +
              method_name +
              ") from " +
              data.from +
              " to target " +
              session_target_id,
          );
        }
      }

      // Check trusted_keys (encryption-based authentication)
      if (this._method_annotations.has(method)) {
        const annotation = this._method_annotations.get(method);
        if (annotation.trusted_keys) {
          if (!callerEncryptionPub) {
            throw new Error(
              `Encryption required for method ${method_name} (trusted_keys is set)`,
            );
          }
          const callerHex = (0,_crypto_js__WEBPACK_IMPORTED_MODULE_2__.publicKeyToHex)(callerEncryptionPub);
          if (!annotation.trusted_keys.has(callerHex)) {
            throw new Error(
              `Caller's public key is not in the trusted keys list for ${method_name}`,
            );
          }
        }
      }

      // Make sure the parent session is still open
      // Skip for service methods — services are persistent and don't
      // depend on the originating session being alive.
      if (local_parent && !data.method.startsWith("services.")) {
        // The parent session should be a session that generate the current method call
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
          this._get_session_store(local_parent, true) !== null,
          "Parent session was closed: " + local_parent,
        );
      }
      let args;
      if (data.args) {
        args = await this._decode(
          data.args,
          data.session,
          null,
          remote_workspace,
          null,
        );
      } else {
        args = [];
      }
      if (
        this._method_annotations.has(method) &&
        this._method_annotations.get(method).require_context
      ) {
        // if args.length + 1 is less than the required number of arguments we will pad with undefined
        // so we make sure the last argument is the context
        if (args.length + 1 < method.length) {
          for (let i = args.length; i < method.length - 1; i++) {
            args.push(undefined);
          }
        }
        args.push(data.ctx);
        // assert(
        //   args.length === method.length,
        //   `Runtime Error: Invalid number of arguments for method ${method_name}, expected ${method.length} but got ${args.length}`,
        // );
      }
      // console.debug(`Executing method: ${method_name} (${data.method})`);
      if (data.promise) {
        const result = method.apply(null, args);
        if (result instanceof Promise) {
          result
            .then((result) => {
              resolve(result);
            })
            .catch((err) => {
              reject(err);
            })
            .finally(() => {
              clearInterval(heartbeat_task);
            });
        } else {
          resolve(result);
          clearInterval(heartbeat_task);
        }
      } else {
        method.apply(null, args);
        clearInterval(heartbeat_task);
      }
    } catch (err) {
      if (reject) {
        reject(err);
      } else {
        console.error("Error during calling method: ", err);
      }
      clearInterval(heartbeat_task);
    }
  }

  encode(aObject, session_id) {
    return this._encode(aObject, session_id);
  }

  _get_session_store(session_id, create) {
    if (!session_id) {
      return null;
    }
    let store = this._object_store;
    const levels = session_id.split(".");
    if (create) {
      const last_index = levels.length - 1;
      for (let level of levels.slice(0, last_index)) {
        if (!store[level]) {
          // Instead of returning null, create intermediate sessions as needed
          store[level] = {};
        }
        store = store[level];
      }
      // Create the last level
      if (!store[levels[last_index]]) {
        store[levels[last_index]] = {};
        const now = Date.now();
        store[levels[last_index]]._created_at = now;
        store[levels[last_index]]._last_activity_at = now;
      }
      return store[levels[last_index]];
    } else {
      for (let level of levels) {
        if (!store[level]) {
          return null;
        }
        store = store[level];
      }
      return store;
    }
  }

  /**
   * Prepares the provided set of remote method arguments for
   * sending to the remote site, replaces all the callbacks with
   * identifiers
   *
   * @param {Array} args to wrap
   *
   * @returns {Array} wrapped arguments
   */
  async _encode(aObject, session_id, local_workspace) {
    const aType = typeof aObject;
    if (
      aType === "number" ||
      aType === "string" ||
      aType === "boolean" ||
      aObject === null ||
      aObject === undefined ||
      aObject instanceof Uint8Array
    ) {
      return aObject;
    }
    if (aObject instanceof ArrayBuffer) {
      return {
        _rtype: "memoryview",
        _rvalue: new Uint8Array(aObject),
      };
    }
    // Reuse the remote object
    if (aObject.__rpc_object__) {
      const _server = aObject.__rpc_object__._rserver || this._server_base_url;
      if (_server === this._server_base_url) {
        return aObject.__rpc_object__;
      } // else {
      //   console.debug(
      //     `Encoding remote function from a different server ${_server}, current server: ${this._server_base_url}`,
      //   );
      // }
    }

    let bObject;

    // skip if already encoded
    if (aObject.constructor instanceof Object && aObject._rtype) {
      // make sure the interface functions are encoded
      const temp = aObject._rtype;
      delete aObject._rtype;
      bObject = await this._encode(aObject, session_id, local_workspace);
      bObject._rtype = temp;
      return bObject;
    }

    if (
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isGenerator)(aObject) ||
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isAsyncGenerator)(aObject) ||
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isAsyncIterator)(aObject) ||
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isSyncIterator)(aObject)
    ) {
      // Handle generator/iterator objects
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        session_id && typeof session_id === "string",
        "Session ID is required for generator encoding",
      );
      const object_id = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
      const close_id = object_id + ":close";

      // Get the session store
      const store = this._get_session_store(session_id, true);
      (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
        store !== null,
        `Failed to create session store ${session_id} due to invalid parent`,
      );

      // Check if it's an async generator/iterator
      const isAsync = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isAsyncGenerator)(aObject) || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.isAsyncIterator)(aObject);

      // Define method to get next item from the generator/iterator
      const nextItemMethod = async () => {
        if (isAsync) {
          const result = await aObject.next();
          if (result.done) {
            delete store[object_id];
            delete store[close_id];
            return { _rtype: "stop_iteration" };
          }
          return result.value;
        } else {
          const result = aObject.next();
          if (result.done) {
            delete store[object_id];
            delete store[close_id];
            return { _rtype: "stop_iteration" };
          }
          return result.value;
        }
      };

      // Define method to close/cleanup the generator/iterator early
      const closeGeneratorMethod = async () => {
        try {
          if (typeof aObject.return === "function") {
            if (isAsync) {
              await aObject.return();
            } else {
              aObject.return();
            }
          }
        } catch (e) {
          // ignore close errors
        } finally {
          delete store[object_id];
          delete store[close_id];
        }
        return true;
      };

      // Store both methods in the session
      store[object_id] = nextItemMethod;
      store[close_id] = closeGeneratorMethod;

      // Create a method that will be used to fetch the next item from the generator
      bObject = {
        _rtype: "generator",
        _rserver: this._server_base_url,
        _rtarget: this._client_id,
        _rmethod: `${session_id}.${object_id}`,
        _rclose_method: `${session_id}.${close_id}`,
        _rpromise: "*",
        _rdoc: "Remote generator",
      };
      return bObject;
    } else if (typeof aObject === "function") {
      if (this._method_annotations.has(aObject)) {
        let annotation = this._method_annotations.get(aObject);
        bObject = {
          _rtype: "method",
          _rserver: this._server_base_url,
          _rtarget: this._client_id,
          _rmethod: annotation.method_id,
          _rpromise: "*",
          _rname: aObject.name,
        };
      } else {
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(typeof session_id === "string");
        let object_id;
        if (aObject.__name__) {
          object_id = `${(0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)()}#${aObject.__name__}`;
        } else {
          object_id = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)();
        }
        bObject = {
          _rtype: "method",
          _rserver: this._server_base_url,
          _rtarget: this._client_id,
          _rmethod: `${session_id}.${object_id}`,
          _rpromise: "*",
          _rname: aObject.name,
        };
        // Attach encryption public key for session methods (e.g. resolve/reject)
        if (this._encryption_enabled) {
          bObject._renc_pub = this._encryption_public_key;
        }
        let store = this._get_session_store(session_id, true);
        (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
          store !== null,
          `Failed to create session store ${session_id} due to invalid parent`,
        );
        store[object_id] = aObject;
      }
      bObject._rdoc = aObject.__doc__;
      if (!bObject._rdoc) {
        try {
          const funcInfo = getFunctionInfo(aObject);
          if (funcInfo && !bObject._rdoc) {
            bObject._rdoc = `${funcInfo.doc}`;
          }
        } catch (e) {
          console.error("Failed to extract function docstring:", aObject);
        }
      }
      bObject._rschema = aObject.__schema__;
      return bObject;
    }
    const isarray = Array.isArray(aObject);

    for (let tp of Object.keys(this._codecs)) {
      const codec = this._codecs[tp];
      if (codec.encoder && aObject instanceof codec.type) {
        // TODO: what if multiple encoders found
        let encodedObj = await Promise.resolve(codec.encoder(aObject));
        if (encodedObj && !encodedObj._rtype) encodedObj._rtype = codec.name;
        // encode the functions in the interface object
        if (typeof encodedObj === "object") {
          const temp = encodedObj._rtype;
          delete encodedObj._rtype;
          encodedObj = await this._encode(
            encodedObj,
            session_id,
            local_workspace,
          );
          encodedObj._rtype = temp;
        }
        bObject = encodedObj;
        return bObject;
      }
    }

    if (
      /*global tf*/
      typeof tf !== "undefined" &&
      tf.Tensor &&
      aObject instanceof tf.Tensor
    ) {
      const v_buffer = aObject.dataSync();
      bObject = {
        _rtype: "ndarray",
        _rvalue: new Uint8Array(v_buffer.buffer),
        _rshape: aObject.shape,
        _rdtype: aObject.dtype,
      };
    } else if (
      /*global nj*/
      typeof nj !== "undefined" &&
      nj.NdArray &&
      aObject instanceof nj.NdArray
    ) {
      if (!aObject.selection || !aObject.selection.data) {
        throw new Error("Invalid NumJS array: missing selection or data");
      }
      const dtype = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.typedArrayToDtype)(aObject.selection.data);
      bObject = {
        _rtype: "ndarray",
        _rvalue: new Uint8Array(aObject.selection.data.buffer),
        _rshape: aObject.shape,
        _rdtype: dtype,
      };
    } else if (aObject instanceof Error) {
      console.error(aObject);
      bObject = {
        _rtype: "error",
        _rvalue: aObject.toString(),
        _rtrace: aObject.stack,
      };
    }
    // send objects supported by structure clone algorithm
    // https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm
    else if (
      aObject !== Object(aObject) ||
      aObject instanceof Boolean ||
      aObject instanceof String ||
      aObject instanceof Date ||
      aObject instanceof RegExp ||
      (typeof ImageData !== "undefined" && aObject instanceof ImageData) ||
      (typeof FileList !== "undefined" && aObject instanceof FileList) ||
      (typeof FileSystemDirectoryHandle !== "undefined" &&
        aObject instanceof FileSystemDirectoryHandle) ||
      (typeof FileSystemFileHandle !== "undefined" &&
        aObject instanceof FileSystemFileHandle) ||
      (typeof FileSystemHandle !== "undefined" &&
        aObject instanceof FileSystemHandle) ||
      (typeof FileSystemWritableFileStream !== "undefined" &&
        aObject instanceof FileSystemWritableFileStream)
    ) {
      bObject = aObject;
      // TODO: avoid object such as DynamicPlugin instance.
    } else if (aObject instanceof Blob) {
      let _current_pos = 0;
      async function read(length) {
        let blob;
        if (length) {
          blob = aObject.slice(_current_pos, _current_pos + length);
        } else {
          blob = aObject.slice(_current_pos);
        }
        const ret = new Uint8Array(await blob.arrayBuffer());
        _current_pos = _current_pos + ret.byteLength;
        return ret;
      }
      function seek(pos) {
        _current_pos = pos;
      }
      bObject = {
        _rtype: "iostream",
        _rnative: "js:blob",
        type: aObject.type,
        name: aObject.name,
        size: aObject.size,
        path: aObject._path || aObject.webkitRelativePath,
        read: await this._encode(read, session_id, local_workspace),
        seek: await this._encode(seek, session_id, local_workspace),
      };
    } else if (aObject instanceof ArrayBufferView) {
      const dtype = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.typedArrayToDtype)(aObject);
      bObject = {
        _rtype: "typedarray",
        _rvalue: new Uint8Array(aObject.buffer),
        _rdtype: dtype,
      };
    } else if (aObject instanceof DataView) {
      bObject = {
        _rtype: "memoryview",
        _rvalue: new Uint8Array(aObject.buffer),
      };
    } else if (aObject instanceof Set) {
      bObject = {
        _rtype: "set",
        _rvalue: await this._encode(
          Array.from(aObject),
          session_id,
          local_workspace,
        ),
      };
    } else if (aObject instanceof Map) {
      bObject = {
        _rtype: "orderedmap",
        _rvalue: await this._encode(
          Array.from(aObject),
          session_id,
          local_workspace,
        ),
      };
    } else if (
      aObject.constructor === Object ||
      Array.isArray(aObject) ||
      aObject instanceof RemoteService
    ) {
      // Auto-register _rintf objects as local services
      if (
        !isarray &&
        aObject._rintf === true &&
        Object.keys(aObject).some(
          (k) => !k.startsWith("_") && typeof aObject[k] === "function",
        )
      ) {
        const serviceId = `_rintf_${(0,_utils_index_js__WEBPACK_IMPORTED_MODULE_0__.randId)()}`;
        const serviceApi = { id: serviceId };
        for (const k of Object.keys(aObject)) {
          if (!k.startsWith("_") && typeof aObject[k] === "function") {
            serviceApi[k] = aObject[k];
          }
        }
        this.add_service(serviceApi, true);
        // Store service_id back on the original object so the caller
        // can later call rpc.unregister_service(serviceId) to clean up.
        aObject._rintf_service_id = serviceId;
        // Encode all values — callables are now annotated as service methods
        bObject = {};
        for (const key of Object.keys(aObject)) {
          bObject[key] = await this._encode(
            aObject[key],
            session_id,
            local_workspace,
          );
        }
        bObject._rintf_service_id = serviceId;
        return bObject;
      }

      // Fast path: if all values are primitives, return as-is
      if (isarray) {
        if (_allPrimitivesArray(aObject)) return aObject;
      } else if (
        !("_rtype" in aObject) &&
        !(aObject instanceof RemoteService)
      ) {
        if (_allPrimitivesObject(aObject)) return aObject;
      }
      bObject = isarray ? [] : {};
      const keys = Object.keys(aObject);
      for (let k of keys) {
        bObject[k] = await this._encode(
          aObject[k],
          session_id,
          local_workspace,
        );
      }
    } else {
      throw `hypha-rpc: Unsupported data type: ${aObject}, you can register a custom codec to encode/decode the object.`;
    }

    if (!bObject) {
      throw new Error("Failed to encode object");
    }
    return bObject;
  }

  async decode(aObject) {
    return await this._decode(aObject);
  }

  async _decode(
    aObject,
    remote_parent,
    local_parent,
    remote_workspace,
    local_workspace,
  ) {
    if (!aObject) {
      return aObject;
    }
    let bObject;
    if (aObject._rtype) {
      if (
        this._codecs[aObject._rtype] &&
        this._codecs[aObject._rtype].decoder
      ) {
        const temp = aObject._rtype;
        delete aObject._rtype;
        aObject = await this._decode(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );
        aObject._rtype = temp;

        bObject = await Promise.resolve(
          this._codecs[aObject._rtype].decoder(aObject),
        );
      } else if (aObject._rtype === "method") {
        bObject = this._generate_remote_method(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );
      } else if (aObject._rtype === "generator") {
        // Create a method to fetch next items from the remote generator
        const gen_method = this._generate_remote_method(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );

        // Create close method if available
        let close_method = null;
        if (aObject._rclose_method) {
          const closeObj = {
            _rtype: "method",
            _rserver: aObject._rserver,
            _rtarget: aObject._rtarget,
            _rmethod: aObject._rclose_method,
            _rpromise: "*",
          };
          close_method = this._generate_remote_method(
            closeObj,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          );
        }

        // Create an async generator proxy with cleanup support
        async function* asyncGeneratorProxy() {
          let completedNormally = false;
          try {
            while (true) {
              const next_item = await gen_method();
              // Check for StopIteration signal
              if (next_item && next_item._rtype === "stop_iteration") {
                completedNormally = true;
                break;
              }
              yield next_item;
            }
          } catch (error) {
            console.error("Error in generator:", error);
            throw error;
          } finally {
            // If not completed normally, send close signal to clean up remote generator
            if (!completedNormally && close_method) {
              try {
                await close_method();
              } catch (e) {
                // ignore close errors
              }
            }
          }
        }
        bObject = asyncGeneratorProxy();
      } else if (aObject._rtype === "ndarray") {
        /*global nj tf*/
        //create build array/tensor if used in the plugin
        if (typeof nj !== "undefined" && nj.array) {
          if (Array.isArray(aObject._rvalue)) {
            aObject._rvalue = aObject._rvalue.reduce(_appendBuffer);
          }
          bObject = nj
            .array(new Uint8(aObject._rvalue), aObject._rdtype)
            .reshape(aObject._rshape);
        } else if (typeof tf !== "undefined" && tf.Tensor) {
          if (Array.isArray(aObject._rvalue)) {
            aObject._rvalue = aObject._rvalue.reduce(_appendBuffer);
          }
          const arraytype = _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.dtypeToTypedArray[aObject._rdtype];
          bObject = tf.tensor(
            new arraytype(aObject._rvalue),
            aObject._rshape,
            aObject._rdtype,
          );
        } else {
          //keep it as regular if transfered to the main app
          bObject = aObject;
        }
      } else if (aObject._rtype === "error") {
        bObject = new Error(
          "RemoteError: " + aObject._rvalue + "\n" + (aObject._rtrace || ""),
        );
      } else if (aObject._rtype === "typedarray") {
        const arraytype = _utils_index_js__WEBPACK_IMPORTED_MODULE_0__.dtypeToTypedArray[aObject._rdtype];
        if (!arraytype)
          throw new Error("unsupported dtype: " + aObject._rdtype);
        const buffer = aObject._rvalue.buffer.slice(
          aObject._rvalue.byteOffset,
          aObject._rvalue.byteOffset + aObject._rvalue.byteLength,
        );
        bObject = new arraytype(buffer);
      } else if (aObject._rtype === "memoryview") {
        bObject = aObject._rvalue.buffer.slice(
          aObject._rvalue.byteOffset,
          aObject._rvalue.byteOffset + aObject._rvalue.byteLength,
        ); // ArrayBuffer
      } else if (aObject._rtype === "iostream") {
        if (aObject._rnative === "js:blob") {
          const read = await this._generate_remote_method(
            aObject.read,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          );
          const bytes = await read();
          bObject = new Blob([bytes], {
            type: aObject.type,
            name: aObject.name,
          });
        } else {
          bObject = {};
          for (let k of Object.keys(aObject)) {
            if (!k.startsWith("_")) {
              bObject[k] = await this._decode(
                aObject[k],
                remote_parent,
                local_parent,
                remote_workspace,
                local_workspace,
              );
            }
          }
        }
        bObject["__rpc_object__"] = aObject;
      } else if (aObject._rtype === "orderedmap") {
        bObject = new Map(
          await this._decode(
            aObject._rvalue,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          ),
        );
      } else if (aObject._rtype === "set") {
        bObject = new Set(
          await this._decode(
            aObject._rvalue,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          ),
        );
      } else {
        const temp = aObject._rtype;
        delete aObject._rtype;
        bObject = await this._decode(
          aObject,
          remote_parent,
          local_parent,
          remote_workspace,
          local_workspace,
        );
        bObject._rtype = temp;
      }
    } else if (aObject.constructor === Object || Array.isArray(aObject)) {
      const isarray = Array.isArray(aObject);
      // Fast path: skip recursive descent if all values are primitives
      if (isarray) {
        if (_allPrimitivesArray(aObject)) return aObject;
      } else {
        if (_allPrimitivesObject(aObject)) return aObject;
      }
      bObject = isarray ? [] : {};
      for (let k of Object.keys(aObject)) {
        if (isarray || aObject.hasOwnProperty(k)) {
          const v = aObject[k];
          bObject[k] = await this._decode(
            v,
            remote_parent,
            local_parent,
            remote_workspace,
            local_workspace,
          );
        }
      }
    } else {
      bObject = aObject;
    }
    if (bObject === undefined) {
      throw new Error("Failed to decode object");
    }
    return bObject;
  }

  _expand_promise(data, callerEncryptionPub) {
    const target = data.from.split("/")[1];
    const session = data.session;
    const method = data.method;
    const heartbeat = {
      _rtype: "method",
      _rtarget: target,
      _rmethod: `${session}.heartbeat`,
      _rdoc: `heartbeat callback for method: ${method}`,
    };
    const resolve = {
      _rtype: "method",
      _rtarget: target,
      _rmethod: `${session}.resolve`,
      _rdoc: `resolve callback for method: ${method}`,
    };
    const reject = {
      _rtype: "method",
      _rtarget: target,
      _rmethod: `${session}.reject`,
      _rdoc: `reject callback for method: ${method}`,
    };
    // Attach caller's encryption pubkey so responses are encrypted back
    if (callerEncryptionPub) {
      heartbeat._renc_pub = callerEncryptionPub;
      resolve._renc_pub = callerEncryptionPub;
      reject._renc_pub = callerEncryptionPub;
    }
    return {
      heartbeat,
      resolve,
      reject,
      interval: data.t,
    };
  }
}


/***/ }),

/***/ "./src/utils/index.js":
/*!****************************!*\
  !*** ./src/utils/index.js ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   MessageEmitter: () => (/* binding */ MessageEmitter),
/* harmony export */   Semaphore: () => (/* binding */ Semaphore),
/* harmony export */   assert: () => (/* binding */ assert),
/* harmony export */   cacheRequirements: () => (/* binding */ cacheRequirements),
/* harmony export */   convertCase: () => (/* binding */ convertCase),
/* harmony export */   dtypeToTypedArray: () => (/* binding */ dtypeToTypedArray),
/* harmony export */   expandKwargs: () => (/* binding */ expandKwargs),
/* harmony export */   isAsyncGenerator: () => (/* binding */ isAsyncGenerator),
/* harmony export */   isAsyncIterator: () => (/* binding */ isAsyncIterator),
/* harmony export */   isGenerator: () => (/* binding */ isGenerator),
/* harmony export */   isSyncIterator: () => (/* binding */ isSyncIterator),
/* harmony export */   loadRequirements: () => (/* binding */ loadRequirements),
/* harmony export */   loadRequirementsInWebworker: () => (/* binding */ loadRequirementsInWebworker),
/* harmony export */   loadRequirementsInWindow: () => (/* binding */ loadRequirementsInWindow),
/* harmony export */   normalizeConfig: () => (/* binding */ normalizeConfig),
/* harmony export */   parseServiceUrl: () => (/* binding */ parseServiceUrl),
/* harmony export */   randId: () => (/* binding */ randId),
/* harmony export */   toCamelCase: () => (/* binding */ toCamelCase),
/* harmony export */   toSnakeCase: () => (/* binding */ toSnakeCase),
/* harmony export */   typedArrayToDtype: () => (/* binding */ typedArrayToDtype),
/* harmony export */   typedArrayToDtypeMapping: () => (/* binding */ typedArrayToDtypeMapping),
/* harmony export */   urlJoin: () => (/* binding */ urlJoin),
/* harmony export */   waitFor: () => (/* binding */ waitFor)
/* harmony export */ });
function randId() {
  return Math.random().toString(36).substr(2, 10) + new Date().getTime();
}

function toCamelCase(str) {
  // Check if the string is already in camelCase
  if (!str.includes("_")) {
    return str;
  }
  // Convert from snake_case to camelCase
  return str.replace(/_./g, (match) => match[1].toUpperCase());
}

function toSnakeCase(str) {
  // Convert from camelCase to snake_case
  return str.replace(/([A-Z])/g, "_$1").toLowerCase();
}

function expandKwargs(obj) {
  if (typeof obj !== "object" || obj === null) {
    return obj; // Return the value if obj is not an object
  }

  const newObj = Array.isArray(obj) ? [] : {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const value = obj[key];

      if (typeof value === "function") {
        newObj[key] = (...args) => {
          if (args.length === 0) {
            throw new Error(`Function "${key}" expects at least one argument.`);
          }

          // Check if the last argument is an object
          const lastArg = args[args.length - 1];
          let kwargs = {};

          if (
            typeof lastArg === "object" &&
            lastArg !== null &&
            !Array.isArray(lastArg)
          ) {
            // Extract kwargs from the last argument
            kwargs = { ...lastArg, _rkwarg: true };
            args = args.slice(0, -1); // Remove the last argument from args
          }

          // Call the original function with positional args followed by kwargs
          return value(...args, kwargs);
        };

        // Preserve metadata like __name__ and __schema__
        newObj[key].__name__ = key;
        if (value.__schema__) {
          newObj[key].__schema__ = { ...value.__schema__ };
          newObj[key].__schema__.name = key;
        }
      } else {
        newObj[key] = expandKwargs(value); // Recursively process nested objects
      }
    }
  }

  return newObj;
}

function convertCase(obj, caseType) {
  if (typeof obj !== "object" || obj === null || !caseType) {
    return obj; // Return the value if obj is not an object
  }

  const newObj = Array.isArray(obj) ? [] : {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const value = obj[key];
      const camelKey = toCamelCase(key);
      const snakeKey = toSnakeCase(key);

      if (caseType === "camel") {
        newObj[camelKey] = convertCase(value, caseType);
        if (typeof value === "function") {
          newObj[camelKey].__name__ = camelKey;
          if (value.__schema__) {
            newObj[camelKey].__schema__ = { ...value.__schema__ };
            newObj[camelKey].__schema__.name = camelKey;
          }
        }
      } else if (caseType === "snake") {
        newObj[snakeKey] = convertCase(value, caseType);
        if (typeof value === "function") {
          newObj[snakeKey].__name__ = snakeKey;
          if (value.__schema__) {
            newObj[snakeKey].__schema__ = { ...value.__schema__ };
            newObj[snakeKey].__schema__.name = snakeKey;
          }
        }
      } else {
        // TODO handle schema for camel + snake
        if (caseType.includes("camel")) {
          newObj[camelKey] = convertCase(value, "camel");
        }
        if (caseType.includes("snake")) {
          newObj[snakeKey] = convertCase(value, "snake");
        }
      }
    }
  }

  return newObj;
}

function parseServiceUrl(url) {
  // Ensure no trailing slash
  url = url.replace(/\/$/, "");

  // Regex pattern to match the URL structure
  const pattern = new RegExp(
    "^(https?:\\/\\/[^/]+)" + // server_url (http or https followed by domain)
      "\\/([a-z0-9_-]+)" + // workspace (lowercase letters, numbers, - or _)
      "\\/services\\/" + // static part of the URL
      "(?:(?<clientId>[a-zA-Z0-9_-]+):)?" + // optional client_id
      "(?<serviceId>[a-zA-Z0-9_-]+)" + // service_id
      "(?:@(?<appId>[a-zA-Z0-9_-]+))?", // optional app_id
  );

  const match = url.match(pattern);
  if (!match) {
    throw new Error("URL does not match the expected pattern");
  }

  const serverUrl = match[1];
  const workspace = match[2];
  const clientId = match.groups?.clientId || "*";
  const serviceId = match.groups?.serviceId;
  const appId = match.groups?.appId || "*";

  return { serverUrl, workspace, clientId, serviceId, appId };
}

const dtypeToTypedArray = {
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  float32: Float32Array,
  float64: Float64Array,
  array: Array,
};

async function loadRequirementsInWindow(requirements) {
  function _importScript(url) {
    //url is URL of external file, implementationCode is the code
    //to be called from the file, location is the location to
    //insert the <script> element
    return new Promise((resolve, reject) => {
      var scriptTag = document.createElement("script");
      scriptTag.src = url;
      scriptTag.type = "text/javascript";
      scriptTag.onload = resolve;
      scriptTag.onreadystatechange = function () {
        if (this.readyState === "loaded" || this.readyState === "complete") {
          resolve();
        }
      };
      scriptTag.onerror = reject;
      document.head.appendChild(scriptTag);
    });
  }

  // support importScripts outside web worker
  async function importScripts() {
    var args = Array.prototype.slice.call(arguments),
      len = args.length,
      i = 0;
    for (; i < len; i++) {
      await _importScript(args[i]);
    }
  }

  if (
    requirements &&
    (Array.isArray(requirements) || typeof requirements === "string")
  ) {
    try {
      var link_node;
      requirements =
        typeof requirements === "string" ? [requirements] : requirements;
      if (Array.isArray(requirements)) {
        for (var i = 0; i < requirements.length; i++) {
          if (
            requirements[i].toLowerCase().endsWith(".css") ||
            requirements[i].startsWith("css:")
          ) {
            if (requirements[i].startsWith("css:")) {
              requirements[i] = requirements[i].slice(4);
            }
            link_node = document.createElement("link");
            link_node.rel = "stylesheet";
            link_node.href = requirements[i];
            document.head.appendChild(link_node);
          } else if (
            requirements[i].toLowerCase().endsWith(".mjs") ||
            requirements[i].startsWith("mjs:")
          ) {
            // import esmodule
            if (requirements[i].startsWith("mjs:")) {
              requirements[i] = requirements[i].slice(4);
            }
            await new Function("url", "return import(url)")(requirements[i]);
          } else if (
            requirements[i].toLowerCase().endsWith(".js") ||
            requirements[i].startsWith("js:")
          ) {
            if (requirements[i].startsWith("js:")) {
              requirements[i] = requirements[i].slice(3);
            }
            await importScripts(requirements[i]);
          } else if (requirements[i].startsWith("http")) {
            await importScripts(requirements[i]);
          } else if (requirements[i].startsWith("cache:")) {
            //ignore cache
          } else {
            console.log("Unprocessed requirements url: " + requirements[i]);
          }
        }
      } else {
        throw "unsupported requirements definition";
      }
    } catch (e) {
      throw "failed to import required scripts: " + requirements.toString();
    }
  }
}

async function loadRequirementsInWebworker(requirements) {
  if (
    requirements &&
    (Array.isArray(requirements) || typeof requirements === "string")
  ) {
    try {
      if (!Array.isArray(requirements)) {
        requirements = [requirements];
      }
      for (var i = 0; i < requirements.length; i++) {
        if (
          requirements[i].toLowerCase().endsWith(".css") ||
          requirements[i].startsWith("css:")
        ) {
          throw "unable to import css in a webworker";
        } else if (
          requirements[i].toLowerCase().endsWith(".js") ||
          requirements[i].startsWith("js:")
        ) {
          if (requirements[i].startsWith("js:")) {
            requirements[i] = requirements[i].slice(3);
          }
          importScripts(requirements[i]);
        } else if (requirements[i].startsWith("http")) {
          importScripts(requirements[i]);
        } else if (requirements[i].startsWith("cache:")) {
          //ignore cache
        } else {
          console.log("Unprocessed requirements url: " + requirements[i]);
        }
      }
    } catch (e) {
      throw "failed to import required scripts: " + requirements.toString();
    }
  }
}

function loadRequirements(requirements) {
  if (
    typeof WorkerGlobalScope !== "undefined" &&
    self instanceof WorkerGlobalScope
  ) {
    return loadRequirementsInWebworker(requirements);
  } else {
    return loadRequirementsInWindow(requirements);
  }
}

function normalizeConfig(config) {
  config.version = config.version || "0.1.0";
  config.description =
    config.description || `[TODO: add description for ${config.name} ]`;
  config.type = config.type || "rpc-window";
  config.id = config.id || randId();
  config.target_origin = config.target_origin || "*";
  config.allow_execution = config.allow_execution || false;
  // remove functions
  config = Object.keys(config).reduce((p, c) => {
    if (typeof config[c] !== "function") p[c] = config[c];
    return p;
  }, {});
  return config;
}
const typedArrayToDtypeMapping = {
  Int8Array: "int8",
  Int16Array: "int16",
  Int32Array: "int32",
  Uint8Array: "uint8",
  Uint16Array: "uint16",
  Uint32Array: "uint32",
  Float32Array: "float32",
  Float64Array: "float64",
  Array: "array",
};

const typedArrayToDtypeKeys = [];
for (const arrType of Object.keys(typedArrayToDtypeMapping)) {
  typedArrayToDtypeKeys.push(eval(arrType));
}

function typedArrayToDtype(obj) {
  let dtype = typedArrayToDtypeMapping[obj.constructor.name];
  if (!dtype) {
    const pt = Object.getPrototypeOf(obj);
    for (const arrType of typedArrayToDtypeKeys) {
      if (pt instanceof arrType) {
        dtype = typedArrayToDtypeMapping[arrType.name];
        break;
      }
    }
  }
  return dtype;
}

function cacheUrlInServiceWorker(url) {
  return new Promise(function (resolve, reject) {
    const message = {
      command: "add",
      url: url,
    };
    if (!navigator.serviceWorker || !navigator.serviceWorker.register) {
      reject("Service worker is not supported.");
      return;
    }
    const messageChannel = new MessageChannel();
    messageChannel.port1.onmessage = function (event) {
      if (event.data && event.data.error) {
        reject(event.data.error);
      } else {
        resolve(event.data && event.data.result);
      }
    };

    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage(message, [
        messageChannel.port2,
      ]);
    } else {
      reject("Service worker controller is not available");
    }
  });
}

async function cacheRequirements(requirements) {
  requirements = requirements || [];
  if (!Array.isArray(requirements)) {
    requirements = [requirements];
  }
  for (let req of requirements) {
    //remove prefix
    if (req.startsWith("js:")) req = req.slice(3);
    if (req.startsWith("css:")) req = req.slice(4);
    if (req.startsWith("cache:")) req = req.slice(6);
    if (!req.startsWith("http")) continue;

    await cacheUrlInServiceWorker(req).catch((e) => {
      console.error(e);
    });
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message || "Assertion failed");
  }
}

//#Source https://bit.ly/2neWfJ2
function urlJoin(...args) {
  return args
    .join("/")
    .replace(/[\/]+/g, "/")
    .replace(/^(.+):\//, "$1://")
    .replace(/^file:/, "file:/")
    .replace(/\/(\?|&|#[^!])/g, "$1")
    .replace(/\?/g, "&")
    .replace("&", "?");
}

function waitFor(prom, time, error) {
  let timer;
  return Promise.race([
    prom,
    new Promise(
      (_r, rej) =>
        (timer = setTimeout(() => {
          rej(error || "Timeout Error");
        }, time * 1000)),
    ),
  ]).finally(() => clearTimeout(timer));
}

class MessageEmitter {
  constructor(debug) {
    this._event_handlers = {};
    this._once_handlers = {};
    this._debug = debug;
  }
  emit() {
    throw new Error("emit is not implemented");
  }
  on(event, handler) {
    if (!this._event_handlers[event]) {
      this._event_handlers[event] = [];
    }
    this._event_handlers[event].push(handler);
  }
  once(event, handler) {
    handler.___event_run_once = true;
    this.on(event, handler);
  }
  off(event, handler) {
    if (!event && !handler) {
      // remove all events handlers
      this._event_handlers = {};
    } else if (event && !handler) {
      // remove all hanlders for the event
      if (this._event_handlers[event]) this._event_handlers[event] = [];
    } else {
      // remove a specific handler
      if (this._event_handlers[event]) {
        const idx = this._event_handlers[event].indexOf(handler);
        if (idx >= 0) {
          this._event_handlers[event].splice(idx, 1);
        }
      }
    }
  }
  _fire(event, data) {
    if (this._event_handlers[event]) {
      var i = this._event_handlers[event].length;
      while (i--) {
        const handler = this._event_handlers[event][i];
        try {
          handler(data);
        } catch (e) {
          console.error(e);
        } finally {
          if (handler.___event_run_once) {
            this._event_handlers[event].splice(i, 1);
          }
        }
      }
    } else {
      if (this._debug) {
        console.warn("unhandled event", event, data);
      }
    }
  }

  waitFor(event, timeout) {
    return new Promise((resolve, reject) => {
      const handler = (data) => {
        clearTimeout(timer);
        resolve(data);
      };
      this.once(event, handler);
      const timer = setTimeout(() => {
        this.off(event, handler);
        reject(new Error("Timeout"));
      }, timeout * 1000);
    });
  }
}

class Semaphore {
  constructor(max) {
    this.max = max;
    this.queue = [];
    this.current = 0;
  }
  async run(task) {
    if (this.current >= this.max) {
      // Wait until a slot is free
      await new Promise((resolve) => this.queue.push(resolve));
    }
    this.current++;
    try {
      return await task();
    } finally {
      this.current--;
      if (this.queue.length > 0) {
        // release one waiter
        this.queue.shift()();
      }
    }
  }
}

/**
 * Check if the object is a generator
 * @param {Object} obj - Object to check
 * @returns {boolean} - True if the object is a generator
 */
function isGenerator(obj) {
  if (!obj) return false;

  return (
    typeof obj === "object" &&
    typeof obj.next === "function" &&
    typeof obj.throw === "function" &&
    typeof obj.return === "function"
  );
}

/**
 * Check if an object is an async generator object
 * @param {any} obj - Object to check
 * @returns {boolean} True if object is an async generator object
 */
function isAsyncGenerator(obj) {
  if (!obj) return false;
  // Check if it's an async generator object
  return (
    typeof obj === "object" &&
    typeof obj.next === "function" &&
    typeof obj.throw === "function" &&
    typeof obj.return === "function" &&
    Symbol.asyncIterator in Object(obj) &&
    obj[Symbol.toStringTag] === "AsyncGenerator"
  );
}

/**
 * Check if an object is a custom async iterator (has Symbol.asyncIterator and next()) but not an async generator
 * @param {any} obj - Object to check
 * @returns {boolean} True if object is a custom async iterator
 */
function isAsyncIterator(obj) {
  if (!obj || isAsyncGenerator(obj)) return false;
  return (
    typeof obj === "object" &&
    Symbol.asyncIterator in Object(obj) &&
    typeof obj.next === "function"
  );
}

/**
 * Check if an object is a custom sync iterator (has Symbol.iterator and next()) but not a generator
 * @param {any} obj - Object to check
 * @returns {boolean} True if object is a custom sync iterator
 */
function isSyncIterator(obj) {
  if (!obj || isGenerator(obj)) return false;
  return (
    typeof obj === "object" &&
    Symbol.iterator in Object(obj) &&
    typeof obj.next === "function" &&
    !Array.isArray(obj) &&
    typeof obj !== "string"
  );
}


/***/ }),

/***/ "./src/utils/schema.js":
/*!*****************************!*\
  !*** ./src/utils/schema.js ***!
  \*****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   schemaFunction: () => (/* binding */ schemaFunction),
/* harmony export */   z: () => (/* binding */ z)
/* harmony export */ });
/* harmony import */ var _index_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./index.js */ "./src/utils/index.js");


// Schema builder utility inspired by Zod for consistent API with Python
const z = {
  object: (properties) => ({
    type: "object",
    properties,
    required: Object.keys(properties).filter(
      (key) => !properties[key]._optional,
    ),
  }),

  string: () => ({ type: "string", _optional: false }),
  number: () => ({ type: "number", _optional: false }),
  integer: () => ({ type: "integer", _optional: false }),
  boolean: () => ({ type: "boolean", _optional: false }),
  array: (items) => ({ type: "array", items, _optional: false }),

  // Make field optional
  optional: (schema) => ({ ...schema, _optional: true }),
};

// Add description method to schema types
["string", "number", "integer", "boolean", "array"].forEach((type) => {
  z[type] = () => {
    const schema = {
      type: type === "integer" ? "integer" : type,
      _optional: false,
    };
    schema.describe = (description) => ({ ...schema, description });
    return schema;
  };
});

function schemaFunction(
  func,
  { schema_type = "auto", name = null, description = null, parameters = null },
) {
  if (!func || typeof func !== "function") {
    throw Error("func should be a function");
  }
  (0,_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(schema_type === "auto", "schema_type should be auto");

  // If no name provided, try to get it from function
  const funcName = name || func.name;
  (0,_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(funcName, "name should not be null");

  // If parameters is a z.object result, convert it properly
  let processedParameters = parameters;
  if (
    parameters &&
    typeof parameters === "object" &&
    parameters.type === "object"
  ) {
    processedParameters = {
      type: "object",
      properties: parameters.properties || {},
      required: parameters.required || [],
    };

    // Clean up internal _optional flags
    for (const [key, schema] of Object.entries(
      processedParameters.properties,
    )) {
      if (schema._optional !== undefined) {
        delete schema._optional;
      }
    }
  }

  (0,_index_js__WEBPACK_IMPORTED_MODULE_0__.assert)(
    processedParameters && processedParameters.type === "object",
    "parameters should be an object schema",
  );

  func.__schema__ = {
    name: funcName,
    description: description || "",
    parameters: processedParameters,
  };
  return func;
}


/***/ }),

/***/ "./src/webrtc-client.js":
/*!******************************!*\
  !*** ./src/webrtc-client.js ***!
  \******************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   getRTCService: () => (/* binding */ getRTCService),
/* harmony export */   registerRTCService: () => (/* binding */ registerRTCService)
/* harmony export */ });
/* harmony import */ var _rpc_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./rpc.js */ "./src/rpc.js");
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");




class WebRTCConnection {
  constructor(channel) {
    this._data_channel = channel;
    this._handle_message = null;
    this._reconnection_token = null;
    this._handle_disconnected = null;
    this._handle_connected = () => {};
    this.manager_id = null;
    this._data_channel.onopen = async () => {
      this._handle_connected &&
        this._handle_connected({ channel: this._data_channel });
    };
    this._data_channel.onmessage = async (event) => {
      let data = event.data;
      if (data instanceof Blob) {
        data = await data.arrayBuffer();
      }
      if (this._handle_message) {
        this._handle_message(data);
      }
    };
    this._data_channel.onclose = () => {
      if (this._handle_disconnected) this._handle_disconnected("closed");
      console.log("data channel closed");
      this._data_channel = null;
    };
  }

  on_disconnected(handler) {
    this._handle_disconnected = handler;
  }

  on_connected(handler) {
    this._handle_connected = handler;
  }

  on_message(handler) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(handler, "handler is required");
    this._handle_message = handler;
  }

  async emit_message(data) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(this._handle_message, "No handler for message");
    try {
      this._data_channel.send(data);
    } catch (exp) {
      console.error(`Failed to send data, error: ${exp}`);
      throw exp;
    }
  }

  async disconnect(reason) {
    this._data_channel = null;
    console.info(`data channel connection disconnected (${reason})`);
  }
}

async function _setupRPC(config) {
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(config.channel, "No channel provided");
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(config.workspace, "No workspace provided");
  const channel = config.channel;
  const clientId = config.client_id || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();
  const connection = new WebRTCConnection(channel);
  config.context = config.context || {};
  config.context.connection_type = "webrtc";
  config.context.ws = config.workspace;
  const rpc = new _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC(connection, {
    client_id: clientId,
    default_context: config.context,
    name: config.name,
    method_timeout: config.method_timeout || 10.0,
    workspace: config.workspace,
    app_id: config.app_id,
    long_message_chunk_size: config.long_message_chunk_size,
  });
  return rpc;
}

async function _createOffer(params, server, config, onInit, context) {
  config = config || {};
  let offer = new RTCSessionDescription({
    sdp: params.sdp,
    type: params.type,
  });

  let pc = new RTCPeerConnection({
    iceServers: config.ice_servers || [
      { urls: ["stun:stun.l.google.com:19302"] },
    ],
    sdpSemantics: "unified-plan",
  });

  if (server) {
    pc.addEventListener("datachannel", async (event) => {
      const channel = event.channel;
      let ctx = null;
      if (context && context.user) ctx = { user: context.user, ws: context.ws };
      const rpc = await _setupRPC({
        channel: channel,
        client_id: channel.label,
        workspace: server.config.workspace,
        context: ctx,
      });
      // Map all the local services to the webrtc client
      rpc._services = server.rpc._services;
    });
  }

  if (onInit) {
    await onInit(pc);
  }

  await pc.setRemoteDescription(offer);

  let answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);

  // Wait for ICE candidates to be gathered (important for Firefox)
  await new Promise((resolveIce) => {
    if (pc.iceGatheringState === "complete") {
      resolveIce();
    } else {
      pc.addEventListener("icegatheringstatechange", () => {
        if (pc.iceGatheringState === "complete") {
          resolveIce();
        }
      });
      // Don't wait forever for ICE gathering
      setTimeout(resolveIce, 5000);
    }
  });

  return {
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    workspace: server.config.workspace,
  };
}

async function getRTCService(server, service_id, config) {
  config = config || {};
  config.peer_id = config.peer_id || (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.randId)();

  const pc = new RTCPeerConnection({
    iceServers: config.ice_servers || [
      { urls: ["stun:stun.l.google.com:19302"] },
    ],
    sdpSemantics: "unified-plan",
  });

  return new Promise(async (resolve, reject) => {
    let resolved = false;
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        pc.close();
        reject(new Error("WebRTC Connection timeout"));
      }
    }, 30000); // Increase timeout to 30 seconds

    try {
      pc.addEventListener(
        "connectionstatechange",
        () => {
          console.log("WebRTC Connection state: ", pc.connectionState);
          if (pc.connectionState === "failed") {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeout);
              pc.close();
              reject(new Error("WebRTC Connection failed"));
            }
          } else if (pc.connectionState === "closed") {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeout);
              reject(new Error("WebRTC Connection closed"));
            }
          } else if (pc.connectionState === "connected") {
            console.log("WebRTC Connection established successfully");
          }
        },
        false,
      );

      // Add ICE connection state change handler for better debugging
      pc.addEventListener("iceconnectionstatechange", () => {
        console.log("ICE Connection state: ", pc.iceConnectionState);
        if (pc.iceConnectionState === "failed") {
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            pc.close();
            reject(new Error("ICE Connection failed"));
          }
        }
      });

      if (config.on_init) {
        await config.on_init(pc);
        delete config.on_init;
      }

      let channel = pc.createDataChannel(config.peer_id, { ordered: true });
      channel.binaryType = "arraybuffer";

      // Wait for ICE gathering to complete before creating offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE candidates to be gathered (important for Firefox)
      await new Promise((resolveIce) => {
        if (pc.iceGatheringState === "complete") {
          resolveIce();
        } else {
          pc.addEventListener("icegatheringstatechange", () => {
            if (pc.iceGatheringState === "complete") {
              resolveIce();
            }
          });
          // Don't wait forever for ICE gathering
          setTimeout(resolveIce, 5000);
        }
      });

      const svc = await server.getService(service_id);
      const answer = await svc.offer({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      });

      channel.onopen = () => {
        config.channel = channel;
        config.workspace = answer.workspace;
        // Increase timeout for Firefox compatibility
        setTimeout(async () => {
          if (!resolved) {
            try {
              const rpc = await _setupRPC(config);
              pc.rpc = rpc;
              async function get_service(name, ...args) {
                (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
                  !name.includes(":"),
                  "WebRTC service name should not contain ':'",
                );
                (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_1__.assert)(
                  !name.includes("/"),
                  "WebRTC service name should not contain '/'",
                );
                return await rpc.get_remote_service(
                  config.workspace + "/" + config.peer_id + ":" + name,
                  ...args,
                );
              }
              async function disconnect() {
                await rpc.disconnect();
                pc.close();
              }
              pc.getService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(get_service, {
                name: "getService",
                description: "Get a remote service via webrtc",
                parameters: {
                  type: "object",
                  properties: {
                    service_id: {
                      type: "string",
                      description:
                        "Service ID. This should be a service id in the format: 'workspace/service_id', 'workspace/client_id:service_id' or 'workspace/client_id:service_id@app_id'",
                    },
                    config: {
                      type: "object",
                      description: "Options for the service",
                    },
                  },
                  required: ["id"],
                },
              });
              pc.disconnect = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(disconnect, {
                name: "disconnect",
                description: "Disconnect from the webrtc connection via webrtc",
                parameters: { type: "object", properties: {} },
              });
              pc.registerCodec = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_2__.schemaFunction)(rpc.register_codec, {
                name: "registerCodec",
                description: "Register a codec for the webrtc connection",
                parameters: {
                  type: "object",
                  properties: {
                    codec: {
                      type: "object",
                      description: "Codec to register",
                      properties: {
                        name: { type: "string" },
                        type: {},
                        encoder: { type: "function" },
                        decoder: { type: "function" },
                      },
                    },
                  },
                },
              });
              resolved = true;
              clearTimeout(timeout);
              resolve(pc);
            } catch (e) {
              if (!resolved) {
                resolved = true;
                clearTimeout(timeout);
                reject(e);
              }
            }
          }
        }, 1000); // Increase timeout to 1 second for Firefox
      };

      channel.onclose = () => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          reject(new Error("Data channel closed"));
        }
      };

      channel.onerror = (error) => {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          reject(new Error(`Data channel error: ${error}`));
        }
      };

      await pc.setRemoteDescription(
        new RTCSessionDescription({
          sdp: answer.sdp,
          type: answer.type,
        }),
      );
    } catch (e) {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        reject(e);
      }
    }
  });
}

async function registerRTCService(server, service_id, config) {
  config = config || {
    visibility: "protected",
    require_context: true,
  };
  const onInit = config.on_init;
  delete config.on_init;
  return await server.registerService({
    id: service_id,
    config,
    offer: (params, context) =>
      _createOffer(params, server, config, onInit, context),
  });
}




/***/ }),

/***/ "?dba7":
/*!************************!*\
  !*** crypto (ignored) ***!
  \************************/
/***/ (() => {

/* (ignored) */

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs":
/*!*************************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs ***!
  \*************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   CachedKeyDecoder: () => (/* binding */ CachedKeyDecoder)
/* harmony export */ });
/* harmony import */ var _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils/utf8.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs");

var DEFAULT_MAX_KEY_LENGTH = 16;
var DEFAULT_MAX_LENGTH_PER_KEY = 16;
var CachedKeyDecoder = /** @class */ (function () {
    function CachedKeyDecoder(maxKeyLength, maxLengthPerKey) {
        if (maxKeyLength === void 0) { maxKeyLength = DEFAULT_MAX_KEY_LENGTH; }
        if (maxLengthPerKey === void 0) { maxLengthPerKey = DEFAULT_MAX_LENGTH_PER_KEY; }
        this.maxKeyLength = maxKeyLength;
        this.maxLengthPerKey = maxLengthPerKey;
        this.hit = 0;
        this.miss = 0;
        // avoid `new Array(N)`, which makes a sparse array,
        // because a sparse array is typically slower than a non-sparse array.
        this.caches = [];
        for (var i = 0; i < this.maxKeyLength; i++) {
            this.caches.push([]);
        }
    }
    CachedKeyDecoder.prototype.canBeCached = function (byteLength) {
        return byteLength > 0 && byteLength <= this.maxKeyLength;
    };
    CachedKeyDecoder.prototype.find = function (bytes, inputOffset, byteLength) {
        var records = this.caches[byteLength - 1];
        FIND_CHUNK: for (var _i = 0, records_1 = records; _i < records_1.length; _i++) {
            var record = records_1[_i];
            var recordBytes = record.bytes;
            for (var j = 0; j < byteLength; j++) {
                if (recordBytes[j] !== bytes[inputOffset + j]) {
                    continue FIND_CHUNK;
                }
            }
            return record.str;
        }
        return null;
    };
    CachedKeyDecoder.prototype.store = function (bytes, value) {
        var records = this.caches[bytes.length - 1];
        var record = { bytes: bytes, str: value };
        if (records.length >= this.maxLengthPerKey) {
            // `records` are full!
            // Set `record` to an arbitrary position.
            records[(Math.random() * records.length) | 0] = record;
        }
        else {
            records.push(record);
        }
    };
    CachedKeyDecoder.prototype.decode = function (bytes, inputOffset, byteLength) {
        var cachedValue = this.find(bytes, inputOffset, byteLength);
        if (cachedValue != null) {
            this.hit++;
            return cachedValue;
        }
        this.miss++;
        var str = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_0__.utf8DecodeJs)(bytes, inputOffset, byteLength);
        // Ensure to copy a slice of bytes because the byte may be NodeJS Buffer and Buffer#slice() returns a reference to its internal ArrayBuffer.
        var slicedCopyOfBytes = Uint8Array.prototype.slice.call(bytes, inputOffset, inputOffset + byteLength);
        this.store(slicedCopyOfBytes, str);
        return str;
    };
    return CachedKeyDecoder;
}());

//# sourceMappingURL=CachedKeyDecoder.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs":
/*!********************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs ***!
  \********************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   DecodeError: () => (/* binding */ DecodeError)
/* harmony export */ });
var __extends = (undefined && undefined.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var DecodeError = /** @class */ (function (_super) {
    __extends(DecodeError, _super);
    function DecodeError(message) {
        var _this = _super.call(this, message) || this;
        // fix the prototype chain in a cross-platform way
        var proto = Object.create(DecodeError.prototype);
        Object.setPrototypeOf(_this, proto);
        Object.defineProperty(_this, "name", {
            configurable: true,
            enumerable: false,
            value: DecodeError.name,
        });
        return _this;
    }
    return DecodeError;
}(Error));

//# sourceMappingURL=DecodeError.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs":
/*!****************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs ***!
  \****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   DataViewIndexOutOfBoundsError: () => (/* binding */ DataViewIndexOutOfBoundsError),
/* harmony export */   Decoder: () => (/* binding */ Decoder)
/* harmony export */ });
/* harmony import */ var _utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./utils/prettyByte.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs");
/* harmony import */ var _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./ExtensionCodec.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs");
/* harmony import */ var _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
/* harmony import */ var _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./utils/utf8.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs");
/* harmony import */ var _utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./utils/typedArrays.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs");
/* harmony import */ var _CachedKeyDecoder_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./CachedKeyDecoder.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/CachedKeyDecoder.mjs");
/* harmony import */ var _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./DecodeError.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs");
var __awaiter = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (undefined && undefined.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __asyncValues = (undefined && undefined.__asyncValues) || function (o) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var m = o[Symbol.asyncIterator], i;
    return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
    function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
    function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
};
var __await = (undefined && undefined.__await) || function (v) { return this instanceof __await ? (this.v = v, this) : new __await(v); }
var __asyncGenerator = (undefined && undefined.__asyncGenerator) || function (thisArg, _arguments, generator) {
    if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
    var g = generator.apply(thisArg, _arguments || []), i, q = [];
    return i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i;
    function verb(n) { if (g[n]) i[n] = function (v) { return new Promise(function (a, b) { q.push([n, v, a, b]) > 1 || resume(n, v); }); }; }
    function resume(n, v) { try { step(g[n](v)); } catch (e) { settle(q[0][3], e); } }
    function step(r) { r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r); }
    function fulfill(value) { resume("next", value); }
    function reject(value) { resume("throw", value); }
    function settle(f, v) { if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]); }
};







var isValidMapKeyType = function (key) {
    var keyType = typeof key;
    return keyType === "string" || keyType === "number";
};
var HEAD_BYTE_REQUIRED = -1;
var EMPTY_VIEW = new DataView(new ArrayBuffer(0));
var EMPTY_BYTES = new Uint8Array(EMPTY_VIEW.buffer);
// IE11: Hack to support IE11.
// IE11: Drop this hack and just use RangeError when IE11 is obsolete.
var DataViewIndexOutOfBoundsError = (function () {
    try {
        // IE11: The spec says it should throw RangeError,
        // IE11: but in IE11 it throws TypeError.
        EMPTY_VIEW.getInt8(0);
    }
    catch (e) {
        return e.constructor;
    }
    throw new Error("never reached");
})();
var MORE_DATA = new DataViewIndexOutOfBoundsError("Insufficient data");
var sharedCachedKeyDecoder = new _CachedKeyDecoder_mjs__WEBPACK_IMPORTED_MODULE_0__.CachedKeyDecoder();
var Decoder = /** @class */ (function () {
    function Decoder(extensionCodec, context, maxStrLength, maxBinLength, maxArrayLength, maxMapLength, maxExtLength, keyDecoder) {
        if (extensionCodec === void 0) { extensionCodec = _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtensionCodec.defaultCodec; }
        if (context === void 0) { context = undefined; }
        if (maxStrLength === void 0) { maxStrLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxBinLength === void 0) { maxBinLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxArrayLength === void 0) { maxArrayLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxMapLength === void 0) { maxMapLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (maxExtLength === void 0) { maxExtLength = _utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.UINT32_MAX; }
        if (keyDecoder === void 0) { keyDecoder = sharedCachedKeyDecoder; }
        this.extensionCodec = extensionCodec;
        this.context = context;
        this.maxStrLength = maxStrLength;
        this.maxBinLength = maxBinLength;
        this.maxArrayLength = maxArrayLength;
        this.maxMapLength = maxMapLength;
        this.maxExtLength = maxExtLength;
        this.keyDecoder = keyDecoder;
        this.totalPos = 0;
        this.pos = 0;
        this.view = EMPTY_VIEW;
        this.bytes = EMPTY_BYTES;
        this.headByte = HEAD_BYTE_REQUIRED;
        this.stack = [];
    }
    Decoder.prototype.reinitializeState = function () {
        this.totalPos = 0;
        this.headByte = HEAD_BYTE_REQUIRED;
        this.stack.length = 0;
        // view, bytes, and pos will be re-initialized in setBuffer()
    };
    Decoder.prototype.setBuffer = function (buffer) {
        this.bytes = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__.ensureUint8Array)(buffer);
        this.view = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__.createDataView)(this.bytes);
        this.pos = 0;
    };
    Decoder.prototype.appendBuffer = function (buffer) {
        if (this.headByte === HEAD_BYTE_REQUIRED && !this.hasRemaining(1)) {
            this.setBuffer(buffer);
        }
        else {
            var remainingData = this.bytes.subarray(this.pos);
            var newData = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_3__.ensureUint8Array)(buffer);
            // concat remainingData + newData
            var newBuffer = new Uint8Array(remainingData.length + newData.length);
            newBuffer.set(remainingData);
            newBuffer.set(newData, remainingData.length);
            this.setBuffer(newBuffer);
        }
    };
    Decoder.prototype.hasRemaining = function (size) {
        return this.view.byteLength - this.pos >= size;
    };
    Decoder.prototype.createExtraByteError = function (posToShow) {
        var _a = this, view = _a.view, pos = _a.pos;
        return new RangeError("Extra ".concat(view.byteLength - pos, " of ").concat(view.byteLength, " byte(s) found at buffer[").concat(posToShow, "]"));
    };
    /**
     * @throws {@link DecodeError}
     * @throws {@link RangeError}
     */
    Decoder.prototype.decode = function (buffer) {
        this.reinitializeState();
        this.setBuffer(buffer);
        var object = this.doDecodeSync();
        if (this.hasRemaining(1)) {
            throw this.createExtraByteError(this.pos);
        }
        return object;
    };
    Decoder.prototype.decodeMulti = function (buffer) {
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    this.reinitializeState();
                    this.setBuffer(buffer);
                    _a.label = 1;
                case 1:
                    if (!this.hasRemaining(1)) return [3 /*break*/, 3];
                    return [4 /*yield*/, this.doDecodeSync()];
                case 2:
                    _a.sent();
                    return [3 /*break*/, 1];
                case 3: return [2 /*return*/];
            }
        });
    };
    Decoder.prototype.decodeAsync = function (stream) {
        var stream_1, stream_1_1;
        var e_1, _a;
        return __awaiter(this, void 0, void 0, function () {
            var decoded, object, buffer, e_1_1, _b, headByte, pos, totalPos;
            return __generator(this, function (_c) {
                switch (_c.label) {
                    case 0:
                        decoded = false;
                        _c.label = 1;
                    case 1:
                        _c.trys.push([1, 6, 7, 12]);
                        stream_1 = __asyncValues(stream);
                        _c.label = 2;
                    case 2: return [4 /*yield*/, stream_1.next()];
                    case 3:
                        if (!(stream_1_1 = _c.sent(), !stream_1_1.done)) return [3 /*break*/, 5];
                        buffer = stream_1_1.value;
                        if (decoded) {
                            throw this.createExtraByteError(this.totalPos);
                        }
                        this.appendBuffer(buffer);
                        try {
                            object = this.doDecodeSync();
                            decoded = true;
                        }
                        catch (e) {
                            if (!(e instanceof DataViewIndexOutOfBoundsError)) {
                                throw e; // rethrow
                            }
                            // fallthrough
                        }
                        this.totalPos += this.pos;
                        _c.label = 4;
                    case 4: return [3 /*break*/, 2];
                    case 5: return [3 /*break*/, 12];
                    case 6:
                        e_1_1 = _c.sent();
                        e_1 = { error: e_1_1 };
                        return [3 /*break*/, 12];
                    case 7:
                        _c.trys.push([7, , 10, 11]);
                        if (!(stream_1_1 && !stream_1_1.done && (_a = stream_1.return))) return [3 /*break*/, 9];
                        return [4 /*yield*/, _a.call(stream_1)];
                    case 8:
                        _c.sent();
                        _c.label = 9;
                    case 9: return [3 /*break*/, 11];
                    case 10:
                        if (e_1) throw e_1.error;
                        return [7 /*endfinally*/];
                    case 11: return [7 /*endfinally*/];
                    case 12:
                        if (decoded) {
                            if (this.hasRemaining(1)) {
                                throw this.createExtraByteError(this.totalPos);
                            }
                            return [2 /*return*/, object];
                        }
                        _b = this, headByte = _b.headByte, pos = _b.pos, totalPos = _b.totalPos;
                        throw new RangeError("Insufficient data in parsing ".concat((0,_utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__.prettyByte)(headByte), " at ").concat(totalPos, " (").concat(pos, " in the current buffer)"));
                }
            });
        });
    };
    Decoder.prototype.decodeArrayStream = function (stream) {
        return this.decodeMultiAsync(stream, true);
    };
    Decoder.prototype.decodeStream = function (stream) {
        return this.decodeMultiAsync(stream, false);
    };
    Decoder.prototype.decodeMultiAsync = function (stream, isArray) {
        return __asyncGenerator(this, arguments, function decodeMultiAsync_1() {
            var isArrayHeaderRequired, arrayItemsLeft, stream_2, stream_2_1, buffer, e_2, e_3_1;
            var e_3, _a;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        isArrayHeaderRequired = isArray;
                        arrayItemsLeft = -1;
                        _b.label = 1;
                    case 1:
                        _b.trys.push([1, 13, 14, 19]);
                        stream_2 = __asyncValues(stream);
                        _b.label = 2;
                    case 2: return [4 /*yield*/, __await(stream_2.next())];
                    case 3:
                        if (!(stream_2_1 = _b.sent(), !stream_2_1.done)) return [3 /*break*/, 12];
                        buffer = stream_2_1.value;
                        if (isArray && arrayItemsLeft === 0) {
                            throw this.createExtraByteError(this.totalPos);
                        }
                        this.appendBuffer(buffer);
                        if (isArrayHeaderRequired) {
                            arrayItemsLeft = this.readArraySize();
                            isArrayHeaderRequired = false;
                            this.complete();
                        }
                        _b.label = 4;
                    case 4:
                        _b.trys.push([4, 9, , 10]);
                        _b.label = 5;
                    case 5:
                        if (false) {}
                        return [4 /*yield*/, __await(this.doDecodeSync())];
                    case 6: return [4 /*yield*/, _b.sent()];
                    case 7:
                        _b.sent();
                        if (--arrayItemsLeft === 0) {
                            return [3 /*break*/, 8];
                        }
                        return [3 /*break*/, 5];
                    case 8: return [3 /*break*/, 10];
                    case 9:
                        e_2 = _b.sent();
                        if (!(e_2 instanceof DataViewIndexOutOfBoundsError)) {
                            throw e_2; // rethrow
                        }
                        return [3 /*break*/, 10];
                    case 10:
                        this.totalPos += this.pos;
                        _b.label = 11;
                    case 11: return [3 /*break*/, 2];
                    case 12: return [3 /*break*/, 19];
                    case 13:
                        e_3_1 = _b.sent();
                        e_3 = { error: e_3_1 };
                        return [3 /*break*/, 19];
                    case 14:
                        _b.trys.push([14, , 17, 18]);
                        if (!(stream_2_1 && !stream_2_1.done && (_a = stream_2.return))) return [3 /*break*/, 16];
                        return [4 /*yield*/, __await(_a.call(stream_2))];
                    case 15:
                        _b.sent();
                        _b.label = 16;
                    case 16: return [3 /*break*/, 18];
                    case 17:
                        if (e_3) throw e_3.error;
                        return [7 /*endfinally*/];
                    case 18: return [7 /*endfinally*/];
                    case 19: return [2 /*return*/];
                }
            });
        });
    };
    Decoder.prototype.doDecodeSync = function () {
        DECODE: while (true) {
            var headByte = this.readHeadByte();
            var object = void 0;
            if (headByte >= 0xe0) {
                // negative fixint (111x xxxx) 0xe0 - 0xff
                object = headByte - 0x100;
            }
            else if (headByte < 0xc0) {
                if (headByte < 0x80) {
                    // positive fixint (0xxx xxxx) 0x00 - 0x7f
                    object = headByte;
                }
                else if (headByte < 0x90) {
                    // fixmap (1000 xxxx) 0x80 - 0x8f
                    var size = headByte - 0x80;
                    if (size !== 0) {
                        this.pushMapState(size);
                        this.complete();
                        continue DECODE;
                    }
                    else {
                        object = {};
                    }
                }
                else if (headByte < 0xa0) {
                    // fixarray (1001 xxxx) 0x90 - 0x9f
                    var size = headByte - 0x90;
                    if (size !== 0) {
                        this.pushArrayState(size);
                        this.complete();
                        continue DECODE;
                    }
                    else {
                        object = [];
                    }
                }
                else {
                    // fixstr (101x xxxx) 0xa0 - 0xbf
                    var byteLength = headByte - 0xa0;
                    object = this.decodeUtf8String(byteLength, 0);
                }
            }
            else if (headByte === 0xc0) {
                // nil
                object = null;
            }
            else if (headByte === 0xc2) {
                // false
                object = false;
            }
            else if (headByte === 0xc3) {
                // true
                object = true;
            }
            else if (headByte === 0xca) {
                // float 32
                object = this.readF32();
            }
            else if (headByte === 0xcb) {
                // float 64
                object = this.readF64();
            }
            else if (headByte === 0xcc) {
                // uint 8
                object = this.readU8();
            }
            else if (headByte === 0xcd) {
                // uint 16
                object = this.readU16();
            }
            else if (headByte === 0xce) {
                // uint 32
                object = this.readU32();
            }
            else if (headByte === 0xcf) {
                // uint 64
                object = this.readU64();
            }
            else if (headByte === 0xd0) {
                // int 8
                object = this.readI8();
            }
            else if (headByte === 0xd1) {
                // int 16
                object = this.readI16();
            }
            else if (headByte === 0xd2) {
                // int 32
                object = this.readI32();
            }
            else if (headByte === 0xd3) {
                // int 64
                object = this.readI64();
            }
            else if (headByte === 0xd9) {
                // str 8
                var byteLength = this.lookU8();
                object = this.decodeUtf8String(byteLength, 1);
            }
            else if (headByte === 0xda) {
                // str 16
                var byteLength = this.lookU16();
                object = this.decodeUtf8String(byteLength, 2);
            }
            else if (headByte === 0xdb) {
                // str 32
                var byteLength = this.lookU32();
                object = this.decodeUtf8String(byteLength, 4);
            }
            else if (headByte === 0xdc) {
                // array 16
                var size = this.readU16();
                if (size !== 0) {
                    this.pushArrayState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = [];
                }
            }
            else if (headByte === 0xdd) {
                // array 32
                var size = this.readU32();
                if (size !== 0) {
                    this.pushArrayState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = [];
                }
            }
            else if (headByte === 0xde) {
                // map 16
                var size = this.readU16();
                if (size !== 0) {
                    this.pushMapState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = {};
                }
            }
            else if (headByte === 0xdf) {
                // map 32
                var size = this.readU32();
                if (size !== 0) {
                    this.pushMapState(size);
                    this.complete();
                    continue DECODE;
                }
                else {
                    object = {};
                }
            }
            else if (headByte === 0xc4) {
                // bin 8
                var size = this.lookU8();
                object = this.decodeBinary(size, 1);
            }
            else if (headByte === 0xc5) {
                // bin 16
                var size = this.lookU16();
                object = this.decodeBinary(size, 2);
            }
            else if (headByte === 0xc6) {
                // bin 32
                var size = this.lookU32();
                object = this.decodeBinary(size, 4);
            }
            else if (headByte === 0xd4) {
                // fixext 1
                object = this.decodeExtension(1, 0);
            }
            else if (headByte === 0xd5) {
                // fixext 2
                object = this.decodeExtension(2, 0);
            }
            else if (headByte === 0xd6) {
                // fixext 4
                object = this.decodeExtension(4, 0);
            }
            else if (headByte === 0xd7) {
                // fixext 8
                object = this.decodeExtension(8, 0);
            }
            else if (headByte === 0xd8) {
                // fixext 16
                object = this.decodeExtension(16, 0);
            }
            else if (headByte === 0xc7) {
                // ext 8
                var size = this.lookU8();
                object = this.decodeExtension(size, 1);
            }
            else if (headByte === 0xc8) {
                // ext 16
                var size = this.lookU16();
                object = this.decodeExtension(size, 2);
            }
            else if (headByte === 0xc9) {
                // ext 32
                var size = this.lookU32();
                object = this.decodeExtension(size, 4);
            }
            else {
                throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Unrecognized type byte: ".concat((0,_utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__.prettyByte)(headByte)));
            }
            this.complete();
            var stack = this.stack;
            while (stack.length > 0) {
                // arrays and maps
                var state = stack[stack.length - 1];
                if (state.type === 0 /* State.ARRAY */) {
                    state.array[state.position] = object;
                    state.position++;
                    if (state.position === state.size) {
                        stack.pop();
                        object = state.array;
                    }
                    else {
                        continue DECODE;
                    }
                }
                else if (state.type === 1 /* State.MAP_KEY */) {
                    if (!isValidMapKeyType(object)) {
                        throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("The type of key must be string or number but " + typeof object);
                    }
                    if (object === "__proto__") {
                        throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("The key __proto__ is not allowed");
                    }
                    state.key = object;
                    state.type = 2 /* State.MAP_VALUE */;
                    continue DECODE;
                }
                else {
                    // it must be `state.type === State.MAP_VALUE` here
                    state.map[state.key] = object;
                    state.readCount++;
                    if (state.readCount === state.size) {
                        stack.pop();
                        object = state.map;
                    }
                    else {
                        state.key = null;
                        state.type = 1 /* State.MAP_KEY */;
                        continue DECODE;
                    }
                }
            }
            return object;
        }
    };
    Decoder.prototype.readHeadByte = function () {
        if (this.headByte === HEAD_BYTE_REQUIRED) {
            this.headByte = this.readU8();
            // console.log("headByte", prettyByte(this.headByte));
        }
        return this.headByte;
    };
    Decoder.prototype.complete = function () {
        this.headByte = HEAD_BYTE_REQUIRED;
    };
    Decoder.prototype.readArraySize = function () {
        var headByte = this.readHeadByte();
        switch (headByte) {
            case 0xdc:
                return this.readU16();
            case 0xdd:
                return this.readU32();
            default: {
                if (headByte < 0xa0) {
                    return headByte - 0x90;
                }
                else {
                    throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Unrecognized array type byte: ".concat((0,_utils_prettyByte_mjs__WEBPACK_IMPORTED_MODULE_4__.prettyByte)(headByte)));
                }
            }
        }
    };
    Decoder.prototype.pushMapState = function (size) {
        if (size > this.maxMapLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: map length (".concat(size, ") > maxMapLengthLength (").concat(this.maxMapLength, ")"));
        }
        this.stack.push({
            type: 1 /* State.MAP_KEY */,
            size: size,
            key: null,
            readCount: 0,
            map: {},
        });
    };
    Decoder.prototype.pushArrayState = function (size) {
        if (size > this.maxArrayLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: array length (".concat(size, ") > maxArrayLength (").concat(this.maxArrayLength, ")"));
        }
        this.stack.push({
            type: 0 /* State.ARRAY */,
            size: size,
            array: new Array(size),
            position: 0,
        });
    };
    Decoder.prototype.decodeUtf8String = function (byteLength, headerOffset) {
        var _a;
        if (byteLength > this.maxStrLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: UTF-8 byte length (".concat(byteLength, ") > maxStrLength (").concat(this.maxStrLength, ")"));
        }
        if (this.bytes.byteLength < this.pos + headerOffset + byteLength) {
            throw MORE_DATA;
        }
        var offset = this.pos + headerOffset;
        var object;
        if (this.stateIsMapKey() && ((_a = this.keyDecoder) === null || _a === void 0 ? void 0 : _a.canBeCached(byteLength))) {
            object = this.keyDecoder.decode(this.bytes, offset, byteLength);
        }
        else if (byteLength > _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__.TEXT_DECODER_THRESHOLD) {
            object = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__.utf8DecodeTD)(this.bytes, offset, byteLength);
        }
        else {
            object = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_6__.utf8DecodeJs)(this.bytes, offset, byteLength);
        }
        this.pos += headerOffset + byteLength;
        return object;
    };
    Decoder.prototype.stateIsMapKey = function () {
        if (this.stack.length > 0) {
            var state = this.stack[this.stack.length - 1];
            return state.type === 1 /* State.MAP_KEY */;
        }
        return false;
    };
    Decoder.prototype.decodeBinary = function (byteLength, headOffset) {
        if (byteLength > this.maxBinLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: bin length (".concat(byteLength, ") > maxBinLength (").concat(this.maxBinLength, ")"));
        }
        if (!this.hasRemaining(byteLength + headOffset)) {
            throw MORE_DATA;
        }
        var offset = this.pos + headOffset;
        var object = this.bytes.subarray(offset, offset + byteLength);
        this.pos += headOffset + byteLength;
        return object;
    };
    Decoder.prototype.decodeExtension = function (size, headOffset) {
        if (size > this.maxExtLength) {
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_5__.DecodeError("Max length exceeded: ext length (".concat(size, ") > maxExtLength (").concat(this.maxExtLength, ")"));
        }
        var extType = this.view.getInt8(this.pos + headOffset);
        var data = this.decodeBinary(size, headOffset + 1 /* extType */);
        return this.extensionCodec.decode(data, extType, this.context);
    };
    Decoder.prototype.lookU8 = function () {
        return this.view.getUint8(this.pos);
    };
    Decoder.prototype.lookU16 = function () {
        return this.view.getUint16(this.pos);
    };
    Decoder.prototype.lookU32 = function () {
        return this.view.getUint32(this.pos);
    };
    Decoder.prototype.readU8 = function () {
        var value = this.view.getUint8(this.pos);
        this.pos++;
        return value;
    };
    Decoder.prototype.readI8 = function () {
        var value = this.view.getInt8(this.pos);
        this.pos++;
        return value;
    };
    Decoder.prototype.readU16 = function () {
        var value = this.view.getUint16(this.pos);
        this.pos += 2;
        return value;
    };
    Decoder.prototype.readI16 = function () {
        var value = this.view.getInt16(this.pos);
        this.pos += 2;
        return value;
    };
    Decoder.prototype.readU32 = function () {
        var value = this.view.getUint32(this.pos);
        this.pos += 4;
        return value;
    };
    Decoder.prototype.readI32 = function () {
        var value = this.view.getInt32(this.pos);
        this.pos += 4;
        return value;
    };
    Decoder.prototype.readU64 = function () {
        var value = (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.getUint64)(this.view, this.pos);
        this.pos += 8;
        return value;
    };
    Decoder.prototype.readI64 = function () {
        var value = (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_2__.getInt64)(this.view, this.pos);
        this.pos += 8;
        return value;
    };
    Decoder.prototype.readF32 = function () {
        var value = this.view.getFloat32(this.pos);
        this.pos += 4;
        return value;
    };
    Decoder.prototype.readF64 = function () {
        var value = this.view.getFloat64(this.pos);
        this.pos += 8;
        return value;
    };
    return Decoder;
}());

//# sourceMappingURL=Decoder.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/Encoder.mjs":
/*!****************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/Encoder.mjs ***!
  \****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   DEFAULT_INITIAL_BUFFER_SIZE: () => (/* binding */ DEFAULT_INITIAL_BUFFER_SIZE),
/* harmony export */   DEFAULT_MAX_DEPTH: () => (/* binding */ DEFAULT_MAX_DEPTH),
/* harmony export */   Encoder: () => (/* binding */ Encoder)
/* harmony export */ });
/* harmony import */ var _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./utils/utf8.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs");
/* harmony import */ var _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./ExtensionCodec.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs");
/* harmony import */ var _utils_int_mjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./utils/int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
/* harmony import */ var _utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/typedArrays.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs");




var DEFAULT_MAX_DEPTH = 100;
var DEFAULT_INITIAL_BUFFER_SIZE = 2048;
var Encoder = /** @class */ (function () {
    function Encoder(extensionCodec, context, maxDepth, initialBufferSize, sortKeys, forceFloat32, ignoreUndefined, forceIntegerToFloat) {
        if (extensionCodec === void 0) { extensionCodec = _ExtensionCodec_mjs__WEBPACK_IMPORTED_MODULE_0__.ExtensionCodec.defaultCodec; }
        if (context === void 0) { context = undefined; }
        if (maxDepth === void 0) { maxDepth = DEFAULT_MAX_DEPTH; }
        if (initialBufferSize === void 0) { initialBufferSize = DEFAULT_INITIAL_BUFFER_SIZE; }
        if (sortKeys === void 0) { sortKeys = false; }
        if (forceFloat32 === void 0) { forceFloat32 = false; }
        if (ignoreUndefined === void 0) { ignoreUndefined = false; }
        if (forceIntegerToFloat === void 0) { forceIntegerToFloat = false; }
        this.extensionCodec = extensionCodec;
        this.context = context;
        this.maxDepth = maxDepth;
        this.initialBufferSize = initialBufferSize;
        this.sortKeys = sortKeys;
        this.forceFloat32 = forceFloat32;
        this.ignoreUndefined = ignoreUndefined;
        this.forceIntegerToFloat = forceIntegerToFloat;
        this.pos = 0;
        this.view = new DataView(new ArrayBuffer(this.initialBufferSize));
        this.bytes = new Uint8Array(this.view.buffer);
    }
    Encoder.prototype.reinitializeState = function () {
        this.pos = 0;
    };
    /**
     * This is almost equivalent to {@link Encoder#encode}, but it returns an reference of the encoder's internal buffer and thus much faster than {@link Encoder#encode}.
     *
     * @returns Encodes the object and returns a shared reference the encoder's internal buffer.
     */
    Encoder.prototype.encodeSharedRef = function (object) {
        this.reinitializeState();
        this.doEncode(object, 1);
        return this.bytes.subarray(0, this.pos);
    };
    /**
     * @returns Encodes the object and returns a copy of the encoder's internal buffer.
     */
    Encoder.prototype.encode = function (object) {
        this.reinitializeState();
        this.doEncode(object, 1);
        return this.bytes.slice(0, this.pos);
    };
    Encoder.prototype.doEncode = function (object, depth) {
        if (depth > this.maxDepth) {
            throw new Error("Too deep objects in depth ".concat(depth));
        }
        if (object == null) {
            this.encodeNil();
        }
        else if (typeof object === "boolean") {
            this.encodeBoolean(object);
        }
        else if (typeof object === "number") {
            this.encodeNumber(object);
        }
        else if (typeof object === "string") {
            this.encodeString(object);
        }
        else {
            this.encodeObject(object, depth);
        }
    };
    Encoder.prototype.ensureBufferSizeToWrite = function (sizeToWrite) {
        var requiredSize = this.pos + sizeToWrite;
        if (this.view.byteLength < requiredSize) {
            this.resizeBuffer(requiredSize * 2);
        }
    };
    Encoder.prototype.resizeBuffer = function (newSize) {
        var newBuffer = new ArrayBuffer(newSize);
        var newBytes = new Uint8Array(newBuffer);
        var newView = new DataView(newBuffer);
        newBytes.set(this.bytes);
        this.view = newView;
        this.bytes = newBytes;
    };
    Encoder.prototype.encodeNil = function () {
        this.writeU8(0xc0);
    };
    Encoder.prototype.encodeBoolean = function (object) {
        if (object === false) {
            this.writeU8(0xc2);
        }
        else {
            this.writeU8(0xc3);
        }
    };
    Encoder.prototype.encodeNumber = function (object) {
        if (Number.isSafeInteger(object) && !this.forceIntegerToFloat) {
            if (object >= 0) {
                if (object < 0x80) {
                    // positive fixint
                    this.writeU8(object);
                }
                else if (object < 0x100) {
                    // uint 8
                    this.writeU8(0xcc);
                    this.writeU8(object);
                }
                else if (object < 0x10000) {
                    // uint 16
                    this.writeU8(0xcd);
                    this.writeU16(object);
                }
                else if (object < 0x100000000) {
                    // uint 32
                    this.writeU8(0xce);
                    this.writeU32(object);
                }
                else {
                    // uint 64
                    this.writeU8(0xcf);
                    this.writeU64(object);
                }
            }
            else {
                if (object >= -0x20) {
                    // negative fixint
                    this.writeU8(0xe0 | (object + 0x20));
                }
                else if (object >= -0x80) {
                    // int 8
                    this.writeU8(0xd0);
                    this.writeI8(object);
                }
                else if (object >= -0x8000) {
                    // int 16
                    this.writeU8(0xd1);
                    this.writeI16(object);
                }
                else if (object >= -0x80000000) {
                    // int 32
                    this.writeU8(0xd2);
                    this.writeI32(object);
                }
                else {
                    // int 64
                    this.writeU8(0xd3);
                    this.writeI64(object);
                }
            }
        }
        else {
            // non-integer numbers
            if (this.forceFloat32) {
                // float 32
                this.writeU8(0xca);
                this.writeF32(object);
            }
            else {
                // float 64
                this.writeU8(0xcb);
                this.writeF64(object);
            }
        }
    };
    Encoder.prototype.writeStringHeader = function (byteLength) {
        if (byteLength < 32) {
            // fixstr
            this.writeU8(0xa0 + byteLength);
        }
        else if (byteLength < 0x100) {
            // str 8
            this.writeU8(0xd9);
            this.writeU8(byteLength);
        }
        else if (byteLength < 0x10000) {
            // str 16
            this.writeU8(0xda);
            this.writeU16(byteLength);
        }
        else if (byteLength < 0x100000000) {
            // str 32
            this.writeU8(0xdb);
            this.writeU32(byteLength);
        }
        else {
            throw new Error("Too long string: ".concat(byteLength, " bytes in UTF-8"));
        }
    };
    Encoder.prototype.encodeString = function (object) {
        var maxHeaderSize = 1 + 4;
        var strLength = object.length;
        if (strLength > _utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.TEXT_ENCODER_THRESHOLD) {
            var byteLength = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8Count)(object);
            this.ensureBufferSizeToWrite(maxHeaderSize + byteLength);
            this.writeStringHeader(byteLength);
            (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8EncodeTE)(object, this.bytes, this.pos);
            this.pos += byteLength;
        }
        else {
            var byteLength = (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8Count)(object);
            this.ensureBufferSizeToWrite(maxHeaderSize + byteLength);
            this.writeStringHeader(byteLength);
            (0,_utils_utf8_mjs__WEBPACK_IMPORTED_MODULE_1__.utf8EncodeJs)(object, this.bytes, this.pos);
            this.pos += byteLength;
        }
    };
    Encoder.prototype.encodeObject = function (object, depth) {
        // try to encode objects with custom codec first of non-primitives
        var ext = this.extensionCodec.tryToEncode(object, this.context);
        if (ext != null) {
            this.encodeExtension(ext);
        }
        else if (Array.isArray(object)) {
            this.encodeArray(object, depth);
        }
        else if (ArrayBuffer.isView(object)) {
            this.encodeBinary(object);
        }
        else if (typeof object === "object") {
            this.encodeMap(object, depth);
        }
        else {
            // symbol, function and other special object come here unless extensionCodec handles them.
            throw new Error("Unrecognized object: ".concat(Object.prototype.toString.apply(object)));
        }
    };
    Encoder.prototype.encodeBinary = function (object) {
        var size = object.byteLength;
        if (size < 0x100) {
            // bin 8
            this.writeU8(0xc4);
            this.writeU8(size);
        }
        else if (size < 0x10000) {
            // bin 16
            this.writeU8(0xc5);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // bin 32
            this.writeU8(0xc6);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large binary: ".concat(size));
        }
        var bytes = (0,_utils_typedArrays_mjs__WEBPACK_IMPORTED_MODULE_2__.ensureUint8Array)(object);
        this.writeU8a(bytes);
    };
    Encoder.prototype.encodeArray = function (object, depth) {
        var size = object.length;
        if (size < 16) {
            // fixarray
            this.writeU8(0x90 + size);
        }
        else if (size < 0x10000) {
            // array 16
            this.writeU8(0xdc);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // array 32
            this.writeU8(0xdd);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large array: ".concat(size));
        }
        for (var _i = 0, object_1 = object; _i < object_1.length; _i++) {
            var item = object_1[_i];
            this.doEncode(item, depth + 1);
        }
    };
    Encoder.prototype.countWithoutUndefined = function (object, keys) {
        var count = 0;
        for (var _i = 0, keys_1 = keys; _i < keys_1.length; _i++) {
            var key = keys_1[_i];
            if (object[key] !== undefined) {
                count++;
            }
        }
        return count;
    };
    Encoder.prototype.encodeMap = function (object, depth) {
        var keys = Object.keys(object);
        if (this.sortKeys) {
            keys.sort();
        }
        var size = this.ignoreUndefined ? this.countWithoutUndefined(object, keys) : keys.length;
        if (size < 16) {
            // fixmap
            this.writeU8(0x80 + size);
        }
        else if (size < 0x10000) {
            // map 16
            this.writeU8(0xde);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // map 32
            this.writeU8(0xdf);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large map object: ".concat(size));
        }
        for (var _i = 0, keys_2 = keys; _i < keys_2.length; _i++) {
            var key = keys_2[_i];
            var value = object[key];
            if (!(this.ignoreUndefined && value === undefined)) {
                this.encodeString(key);
                this.doEncode(value, depth + 1);
            }
        }
    };
    Encoder.prototype.encodeExtension = function (ext) {
        var size = ext.data.length;
        if (size === 1) {
            // fixext 1
            this.writeU8(0xd4);
        }
        else if (size === 2) {
            // fixext 2
            this.writeU8(0xd5);
        }
        else if (size === 4) {
            // fixext 4
            this.writeU8(0xd6);
        }
        else if (size === 8) {
            // fixext 8
            this.writeU8(0xd7);
        }
        else if (size === 16) {
            // fixext 16
            this.writeU8(0xd8);
        }
        else if (size < 0x100) {
            // ext 8
            this.writeU8(0xc7);
            this.writeU8(size);
        }
        else if (size < 0x10000) {
            // ext 16
            this.writeU8(0xc8);
            this.writeU16(size);
        }
        else if (size < 0x100000000) {
            // ext 32
            this.writeU8(0xc9);
            this.writeU32(size);
        }
        else {
            throw new Error("Too large extension object: ".concat(size));
        }
        this.writeI8(ext.type);
        this.writeU8a(ext.data);
    };
    Encoder.prototype.writeU8 = function (value) {
        this.ensureBufferSizeToWrite(1);
        this.view.setUint8(this.pos, value);
        this.pos++;
    };
    Encoder.prototype.writeU8a = function (values) {
        var size = values.length;
        this.ensureBufferSizeToWrite(size);
        this.bytes.set(values, this.pos);
        this.pos += size;
    };
    Encoder.prototype.writeI8 = function (value) {
        this.ensureBufferSizeToWrite(1);
        this.view.setInt8(this.pos, value);
        this.pos++;
    };
    Encoder.prototype.writeU16 = function (value) {
        this.ensureBufferSizeToWrite(2);
        this.view.setUint16(this.pos, value);
        this.pos += 2;
    };
    Encoder.prototype.writeI16 = function (value) {
        this.ensureBufferSizeToWrite(2);
        this.view.setInt16(this.pos, value);
        this.pos += 2;
    };
    Encoder.prototype.writeU32 = function (value) {
        this.ensureBufferSizeToWrite(4);
        this.view.setUint32(this.pos, value);
        this.pos += 4;
    };
    Encoder.prototype.writeI32 = function (value) {
        this.ensureBufferSizeToWrite(4);
        this.view.setInt32(this.pos, value);
        this.pos += 4;
    };
    Encoder.prototype.writeF32 = function (value) {
        this.ensureBufferSizeToWrite(4);
        this.view.setFloat32(this.pos, value);
        this.pos += 4;
    };
    Encoder.prototype.writeF64 = function (value) {
        this.ensureBufferSizeToWrite(8);
        this.view.setFloat64(this.pos, value);
        this.pos += 8;
    };
    Encoder.prototype.writeU64 = function (value) {
        this.ensureBufferSizeToWrite(8);
        (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_3__.setUint64)(this.view, this.pos, value);
        this.pos += 8;
    };
    Encoder.prototype.writeI64 = function (value) {
        this.ensureBufferSizeToWrite(8);
        (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_3__.setInt64)(this.view, this.pos, value);
        this.pos += 8;
    };
    return Encoder;
}());

//# sourceMappingURL=Encoder.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs":
/*!****************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs ***!
  \****************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   ExtData: () => (/* binding */ ExtData)
/* harmony export */ });
/**
 * ExtData is used to handle Extension Types that are not registered to ExtensionCodec.
 */
var ExtData = /** @class */ (function () {
    function ExtData(type, data) {
        this.type = type;
        this.data = data;
    }
    return ExtData;
}());

//# sourceMappingURL=ExtData.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs":
/*!***********************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/ExtensionCodec.mjs ***!
  \***********************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   ExtensionCodec: () => (/* binding */ ExtensionCodec)
/* harmony export */ });
/* harmony import */ var _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./ExtData.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/ExtData.mjs");
/* harmony import */ var _timestamp_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./timestamp.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs");
// ExtensionCodec to handle MessagePack extensions


var ExtensionCodec = /** @class */ (function () {
    function ExtensionCodec() {
        // built-in extensions
        this.builtInEncoders = [];
        this.builtInDecoders = [];
        // custom extensions
        this.encoders = [];
        this.decoders = [];
        this.register(_timestamp_mjs__WEBPACK_IMPORTED_MODULE_0__.timestampExtension);
    }
    ExtensionCodec.prototype.register = function (_a) {
        var type = _a.type, encode = _a.encode, decode = _a.decode;
        if (type >= 0) {
            // custom extensions
            this.encoders[type] = encode;
            this.decoders[type] = decode;
        }
        else {
            // built-in extensions
            var index = 1 + type;
            this.builtInEncoders[index] = encode;
            this.builtInDecoders[index] = decode;
        }
    };
    ExtensionCodec.prototype.tryToEncode = function (object, context) {
        // built-in extensions
        for (var i = 0; i < this.builtInEncoders.length; i++) {
            var encodeExt = this.builtInEncoders[i];
            if (encodeExt != null) {
                var data = encodeExt(object, context);
                if (data != null) {
                    var type = -1 - i;
                    return new _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData(type, data);
                }
            }
        }
        // custom extensions
        for (var i = 0; i < this.encoders.length; i++) {
            var encodeExt = this.encoders[i];
            if (encodeExt != null) {
                var data = encodeExt(object, context);
                if (data != null) {
                    var type = i;
                    return new _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData(type, data);
                }
            }
        }
        if (object instanceof _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData) {
            // to keep ExtData as is
            return object;
        }
        return null;
    };
    ExtensionCodec.prototype.decode = function (data, type, context) {
        var decodeExt = type < 0 ? this.builtInDecoders[-1 - type] : this.decoders[type];
        if (decodeExt) {
            return decodeExt(data, type, context);
        }
        else {
            // decode() does not fail, returns ExtData instead.
            return new _ExtData_mjs__WEBPACK_IMPORTED_MODULE_1__.ExtData(type, data);
        }
    };
    ExtensionCodec.defaultCodec = new ExtensionCodec();
    return ExtensionCodec;
}());

//# sourceMappingURL=ExtensionCodec.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs":
/*!***************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/decode.mjs ***!
  \***************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   decode: () => (/* binding */ decode),
/* harmony export */   decodeMulti: () => (/* binding */ decodeMulti),
/* harmony export */   defaultDecodeOptions: () => (/* binding */ defaultDecodeOptions)
/* harmony export */ });
/* harmony import */ var _Decoder_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Decoder.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/Decoder.mjs");

var defaultDecodeOptions = {};
/**
 * It decodes a single MessagePack object in a buffer.
 *
 * This is a synchronous decoding function.
 * See other variants for asynchronous decoding: {@link decodeAsync()}, {@link decodeStream()}, or {@link decodeArrayStream()}.
 *
 * @throws {@link RangeError} if the buffer is incomplete, including the case where the buffer is empty.
 * @throws {@link DecodeError} if the buffer contains invalid data.
 */
function decode(buffer, options) {
    if (options === void 0) { options = defaultDecodeOptions; }
    var decoder = new _Decoder_mjs__WEBPACK_IMPORTED_MODULE_0__.Decoder(options.extensionCodec, options.context, options.maxStrLength, options.maxBinLength, options.maxArrayLength, options.maxMapLength, options.maxExtLength);
    return decoder.decode(buffer);
}
/**
 * It decodes multiple MessagePack objects in a buffer.
 * This is corresponding to {@link decodeMultiStream()}.
 *
 * @throws {@link RangeError} if the buffer is incomplete, including the case where the buffer is empty.
 * @throws {@link DecodeError} if the buffer contains invalid data.
 */
function decodeMulti(buffer, options) {
    if (options === void 0) { options = defaultDecodeOptions; }
    var decoder = new _Decoder_mjs__WEBPACK_IMPORTED_MODULE_0__.Decoder(options.extensionCodec, options.context, options.maxStrLength, options.maxBinLength, options.maxArrayLength, options.maxMapLength, options.maxExtLength);
    return decoder.decodeMulti(buffer);
}
//# sourceMappingURL=decode.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs":
/*!***************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/encode.mjs ***!
  \***************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   encode: () => (/* binding */ encode)
/* harmony export */ });
/* harmony import */ var _Encoder_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./Encoder.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/Encoder.mjs");

var defaultEncodeOptions = {};
/**
 * It encodes `value` in the MessagePack format and
 * returns a byte buffer.
 *
 * The returned buffer is a slice of a larger `ArrayBuffer`, so you have to use its `#byteOffset` and `#byteLength` in order to convert it to another typed arrays including NodeJS `Buffer`.
 */
function encode(value, options) {
    if (options === void 0) { options = defaultEncodeOptions; }
    var encoder = new _Encoder_mjs__WEBPACK_IMPORTED_MODULE_0__.Encoder(options.extensionCodec, options.context, options.maxDepth, options.initialBufferSize, options.sortKeys, options.forceFloat32, options.ignoreUndefined, options.forceIntegerToFloat);
    return encoder.encodeSharedRef(value);
}
//# sourceMappingURL=encode.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs":
/*!******************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/timestamp.mjs ***!
  \******************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   EXT_TIMESTAMP: () => (/* binding */ EXT_TIMESTAMP),
/* harmony export */   decodeTimestampExtension: () => (/* binding */ decodeTimestampExtension),
/* harmony export */   decodeTimestampToTimeSpec: () => (/* binding */ decodeTimestampToTimeSpec),
/* harmony export */   encodeDateToTimeSpec: () => (/* binding */ encodeDateToTimeSpec),
/* harmony export */   encodeTimeSpecToTimestamp: () => (/* binding */ encodeTimeSpecToTimestamp),
/* harmony export */   encodeTimestampExtension: () => (/* binding */ encodeTimestampExtension),
/* harmony export */   timestampExtension: () => (/* binding */ timestampExtension)
/* harmony export */ });
/* harmony import */ var _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./DecodeError.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/DecodeError.mjs");
/* harmony import */ var _utils_int_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./utils/int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
// https://github.com/msgpack/msgpack/blob/master/spec.md#timestamp-extension-type


var EXT_TIMESTAMP = -1;
var TIMESTAMP32_MAX_SEC = 0x100000000 - 1; // 32-bit unsigned int
var TIMESTAMP64_MAX_SEC = 0x400000000 - 1; // 34-bit unsigned int
function encodeTimeSpecToTimestamp(_a) {
    var sec = _a.sec, nsec = _a.nsec;
    if (sec >= 0 && nsec >= 0 && sec <= TIMESTAMP64_MAX_SEC) {
        // Here sec >= 0 && nsec >= 0
        if (nsec === 0 && sec <= TIMESTAMP32_MAX_SEC) {
            // timestamp 32 = { sec32 (unsigned) }
            var rv = new Uint8Array(4);
            var view = new DataView(rv.buffer);
            view.setUint32(0, sec);
            return rv;
        }
        else {
            // timestamp 64 = { nsec30 (unsigned), sec34 (unsigned) }
            var secHigh = sec / 0x100000000;
            var secLow = sec & 0xffffffff;
            var rv = new Uint8Array(8);
            var view = new DataView(rv.buffer);
            // nsec30 | secHigh2
            view.setUint32(0, (nsec << 2) | (secHigh & 0x3));
            // secLow32
            view.setUint32(4, secLow);
            return rv;
        }
    }
    else {
        // timestamp 96 = { nsec32 (unsigned), sec64 (signed) }
        var rv = new Uint8Array(12);
        var view = new DataView(rv.buffer);
        view.setUint32(0, nsec);
        (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_0__.setInt64)(view, 4, sec);
        return rv;
    }
}
function encodeDateToTimeSpec(date) {
    var msec = date.getTime();
    var sec = Math.floor(msec / 1e3);
    var nsec = (msec - sec * 1e3) * 1e6;
    // Normalizes { sec, nsec } to ensure nsec is unsigned.
    var nsecInSec = Math.floor(nsec / 1e9);
    return {
        sec: sec + nsecInSec,
        nsec: nsec - nsecInSec * 1e9,
    };
}
function encodeTimestampExtension(object) {
    if (object instanceof Date) {
        var timeSpec = encodeDateToTimeSpec(object);
        return encodeTimeSpecToTimestamp(timeSpec);
    }
    else {
        return null;
    }
}
function decodeTimestampToTimeSpec(data) {
    var view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    // data may be 32, 64, or 96 bits
    switch (data.byteLength) {
        case 4: {
            // timestamp 32 = { sec32 }
            var sec = view.getUint32(0);
            var nsec = 0;
            return { sec: sec, nsec: nsec };
        }
        case 8: {
            // timestamp 64 = { nsec30, sec34 }
            var nsec30AndSecHigh2 = view.getUint32(0);
            var secLow32 = view.getUint32(4);
            var sec = (nsec30AndSecHigh2 & 0x3) * 0x100000000 + secLow32;
            var nsec = nsec30AndSecHigh2 >>> 2;
            return { sec: sec, nsec: nsec };
        }
        case 12: {
            // timestamp 96 = { nsec32 (unsigned), sec64 (signed) }
            var sec = (0,_utils_int_mjs__WEBPACK_IMPORTED_MODULE_0__.getInt64)(view, 4);
            var nsec = view.getUint32(0);
            return { sec: sec, nsec: nsec };
        }
        default:
            throw new _DecodeError_mjs__WEBPACK_IMPORTED_MODULE_1__.DecodeError("Unrecognized data size for timestamp (expected 4, 8, or 12): ".concat(data.length));
    }
}
function decodeTimestampExtension(data) {
    var timeSpec = decodeTimestampToTimeSpec(data);
    return new Date(timeSpec.sec * 1e3 + timeSpec.nsec / 1e6);
}
var timestampExtension = {
    type: EXT_TIMESTAMP,
    encode: encodeTimestampExtension,
    decode: decodeTimestampExtension,
};
//# sourceMappingURL=timestamp.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs":
/*!******************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs ***!
  \******************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   UINT32_MAX: () => (/* binding */ UINT32_MAX),
/* harmony export */   getInt64: () => (/* binding */ getInt64),
/* harmony export */   getUint64: () => (/* binding */ getUint64),
/* harmony export */   setInt64: () => (/* binding */ setInt64),
/* harmony export */   setUint64: () => (/* binding */ setUint64)
/* harmony export */ });
// Integer Utility
var UINT32_MAX = 4294967295;
// DataView extension to handle int64 / uint64,
// where the actual range is 53-bits integer (a.k.a. safe integer)
function setUint64(view, offset, value) {
    var high = value / 4294967296;
    var low = value; // high bits are truncated by DataView
    view.setUint32(offset, high);
    view.setUint32(offset + 4, low);
}
function setInt64(view, offset, value) {
    var high = Math.floor(value / 4294967296);
    var low = value; // high bits are truncated by DataView
    view.setUint32(offset, high);
    view.setUint32(offset + 4, low);
}
function getInt64(view, offset) {
    var high = view.getInt32(offset);
    var low = view.getUint32(offset + 4);
    return high * 4294967296 + low;
}
function getUint64(view, offset) {
    var high = view.getUint32(offset);
    var low = view.getUint32(offset + 4);
    return high * 4294967296 + low;
}
//# sourceMappingURL=int.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs":
/*!*************************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/prettyByte.mjs ***!
  \*************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   prettyByte: () => (/* binding */ prettyByte)
/* harmony export */ });
function prettyByte(byte) {
    return "".concat(byte < 0 ? "-" : "", "0x").concat(Math.abs(byte).toString(16).padStart(2, "0"));
}
//# sourceMappingURL=prettyByte.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs":
/*!**************************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/typedArrays.mjs ***!
  \**************************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   createDataView: () => (/* binding */ createDataView),
/* harmony export */   ensureUint8Array: () => (/* binding */ ensureUint8Array)
/* harmony export */ });
function ensureUint8Array(buffer) {
    if (buffer instanceof Uint8Array) {
        return buffer;
    }
    else if (ArrayBuffer.isView(buffer)) {
        return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
    else if (buffer instanceof ArrayBuffer) {
        return new Uint8Array(buffer);
    }
    else {
        // ArrayLike<number>
        return Uint8Array.from(buffer);
    }
}
function createDataView(buffer) {
    if (buffer instanceof ArrayBuffer) {
        return new DataView(buffer);
    }
    var bufferView = ensureUint8Array(buffer);
    return new DataView(bufferView.buffer, bufferView.byteOffset, bufferView.byteLength);
}
//# sourceMappingURL=typedArrays.mjs.map

/***/ }),

/***/ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs":
/*!*******************************************************************!*\
  !*** ./node_modules/@msgpack/msgpack/dist.es5+esm/utils/utf8.mjs ***!
  \*******************************************************************/
/***/ ((__unused_webpack___webpack_module__, __webpack_exports__, __webpack_require__) => {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   TEXT_DECODER_THRESHOLD: () => (/* binding */ TEXT_DECODER_THRESHOLD),
/* harmony export */   TEXT_ENCODER_THRESHOLD: () => (/* binding */ TEXT_ENCODER_THRESHOLD),
/* harmony export */   utf8Count: () => (/* binding */ utf8Count),
/* harmony export */   utf8DecodeJs: () => (/* binding */ utf8DecodeJs),
/* harmony export */   utf8DecodeTD: () => (/* binding */ utf8DecodeTD),
/* harmony export */   utf8EncodeJs: () => (/* binding */ utf8EncodeJs),
/* harmony export */   utf8EncodeTE: () => (/* binding */ utf8EncodeTE)
/* harmony export */ });
/* harmony import */ var _int_mjs__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./int.mjs */ "./node_modules/@msgpack/msgpack/dist.es5+esm/utils/int.mjs");
var _a, _b, _c;
/* eslint-disable @typescript-eslint/no-unnecessary-condition */

var TEXT_ENCODING_AVAILABLE = (typeof process === "undefined" || ((_a = process === null || process === void 0 ? void 0 : process.env) === null || _a === void 0 ? void 0 : _a["TEXT_ENCODING"]) !== "never") &&
    typeof TextEncoder !== "undefined" &&
    typeof TextDecoder !== "undefined";
function utf8Count(str) {
    var strLength = str.length;
    var byteLength = 0;
    var pos = 0;
    while (pos < strLength) {
        var value = str.charCodeAt(pos++);
        if ((value & 0xffffff80) === 0) {
            // 1-byte
            byteLength++;
            continue;
        }
        else if ((value & 0xfffff800) === 0) {
            // 2-bytes
            byteLength += 2;
        }
        else {
            // handle surrogate pair
            if (value >= 0xd800 && value <= 0xdbff) {
                // high surrogate
                if (pos < strLength) {
                    var extra = str.charCodeAt(pos);
                    if ((extra & 0xfc00) === 0xdc00) {
                        ++pos;
                        value = ((value & 0x3ff) << 10) + (extra & 0x3ff) + 0x10000;
                    }
                }
            }
            if ((value & 0xffff0000) === 0) {
                // 3-byte
                byteLength += 3;
            }
            else {
                // 4-byte
                byteLength += 4;
            }
        }
    }
    return byteLength;
}
function utf8EncodeJs(str, output, outputOffset) {
    var strLength = str.length;
    var offset = outputOffset;
    var pos = 0;
    while (pos < strLength) {
        var value = str.charCodeAt(pos++);
        if ((value & 0xffffff80) === 0) {
            // 1-byte
            output[offset++] = value;
            continue;
        }
        else if ((value & 0xfffff800) === 0) {
            // 2-bytes
            output[offset++] = ((value >> 6) & 0x1f) | 0xc0;
        }
        else {
            // handle surrogate pair
            if (value >= 0xd800 && value <= 0xdbff) {
                // high surrogate
                if (pos < strLength) {
                    var extra = str.charCodeAt(pos);
                    if ((extra & 0xfc00) === 0xdc00) {
                        ++pos;
                        value = ((value & 0x3ff) << 10) + (extra & 0x3ff) + 0x10000;
                    }
                }
            }
            if ((value & 0xffff0000) === 0) {
                // 3-byte
                output[offset++] = ((value >> 12) & 0x0f) | 0xe0;
                output[offset++] = ((value >> 6) & 0x3f) | 0x80;
            }
            else {
                // 4-byte
                output[offset++] = ((value >> 18) & 0x07) | 0xf0;
                output[offset++] = ((value >> 12) & 0x3f) | 0x80;
                output[offset++] = ((value >> 6) & 0x3f) | 0x80;
            }
        }
        output[offset++] = (value & 0x3f) | 0x80;
    }
}
var sharedTextEncoder = TEXT_ENCODING_AVAILABLE ? new TextEncoder() : undefined;
var TEXT_ENCODER_THRESHOLD = !TEXT_ENCODING_AVAILABLE
    ? _int_mjs__WEBPACK_IMPORTED_MODULE_0__.UINT32_MAX
    : typeof process !== "undefined" && ((_b = process === null || process === void 0 ? void 0 : process.env) === null || _b === void 0 ? void 0 : _b["TEXT_ENCODING"]) !== "force"
        ? 200
        : 0;
function utf8EncodeTEencode(str, output, outputOffset) {
    output.set(sharedTextEncoder.encode(str), outputOffset);
}
function utf8EncodeTEencodeInto(str, output, outputOffset) {
    sharedTextEncoder.encodeInto(str, output.subarray(outputOffset));
}
var utf8EncodeTE = (sharedTextEncoder === null || sharedTextEncoder === void 0 ? void 0 : sharedTextEncoder.encodeInto) ? utf8EncodeTEencodeInto : utf8EncodeTEencode;
var CHUNK_SIZE = 4096;
function utf8DecodeJs(bytes, inputOffset, byteLength) {
    var offset = inputOffset;
    var end = offset + byteLength;
    var units = [];
    var result = "";
    while (offset < end) {
        var byte1 = bytes[offset++];
        if ((byte1 & 0x80) === 0) {
            // 1 byte
            units.push(byte1);
        }
        else if ((byte1 & 0xe0) === 0xc0) {
            // 2 bytes
            var byte2 = bytes[offset++] & 0x3f;
            units.push(((byte1 & 0x1f) << 6) | byte2);
        }
        else if ((byte1 & 0xf0) === 0xe0) {
            // 3 bytes
            var byte2 = bytes[offset++] & 0x3f;
            var byte3 = bytes[offset++] & 0x3f;
            units.push(((byte1 & 0x1f) << 12) | (byte2 << 6) | byte3);
        }
        else if ((byte1 & 0xf8) === 0xf0) {
            // 4 bytes
            var byte2 = bytes[offset++] & 0x3f;
            var byte3 = bytes[offset++] & 0x3f;
            var byte4 = bytes[offset++] & 0x3f;
            var unit = ((byte1 & 0x07) << 0x12) | (byte2 << 0x0c) | (byte3 << 0x06) | byte4;
            if (unit > 0xffff) {
                unit -= 0x10000;
                units.push(((unit >>> 10) & 0x3ff) | 0xd800);
                unit = 0xdc00 | (unit & 0x3ff);
            }
            units.push(unit);
        }
        else {
            units.push(byte1);
        }
        if (units.length >= CHUNK_SIZE) {
            result += String.fromCharCode.apply(String, units);
            units.length = 0;
        }
    }
    if (units.length > 0) {
        result += String.fromCharCode.apply(String, units);
    }
    return result;
}
var sharedTextDecoder = TEXT_ENCODING_AVAILABLE ? new TextDecoder() : null;
var TEXT_DECODER_THRESHOLD = !TEXT_ENCODING_AVAILABLE
    ? _int_mjs__WEBPACK_IMPORTED_MODULE_0__.UINT32_MAX
    : typeof process !== "undefined" && ((_c = process === null || process === void 0 ? void 0 : process.env) === null || _c === void 0 ? void 0 : _c["TEXT_DECODER"]) !== "force"
        ? 200
        : 0;
function utf8DecodeTD(bytes, inputOffset, byteLength) {
    var stringBytes = bytes.subarray(inputOffset, inputOffset + byteLength);
    return sharedTextDecoder.decode(stringBytes);
}
//# sourceMappingURL=utf8.mjs.map

/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/define property getters */
/******/ 	(() => {
/******/ 		// define getter functions for harmony exports
/******/ 		__webpack_require__.d = (exports, definition) => {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/hasOwnProperty shorthand */
/******/ 	(() => {
/******/ 		__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ 	})();
/******/ 	
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
// This entry need to be wrapped in an IIFE because it need to be in strict mode.
(() => {
"use strict";
/*!*********************************!*\
  !*** ./src/websocket-client.js ***!
  \*********************************/
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   API_VERSION: () => (/* reexport safe */ _rpc_js__WEBPACK_IMPORTED_MODULE_0__.API_VERSION),
/* harmony export */   HTTPStreamingRPCConnection: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_5__.HTTPStreamingRPCConnection),
/* harmony export */   LocalWebSocket: () => (/* binding */ LocalWebSocket),
/* harmony export */   RPC: () => (/* reexport safe */ _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC),
/* harmony export */   connectToServer: () => (/* binding */ connectToServer),
/* harmony export */   connectToServerHTTP: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_5__.connectToServerHTTP),
/* harmony export */   decryptPayload: () => (/* reexport safe */ _crypto_js__WEBPACK_IMPORTED_MODULE_1__.decryptPayload),
/* harmony export */   encryptPayload: () => (/* reexport safe */ _crypto_js__WEBPACK_IMPORTED_MODULE_1__.encryptPayload),
/* harmony export */   generateEncryptionKeypair: () => (/* reexport safe */ _crypto_js__WEBPACK_IMPORTED_MODULE_1__.generateEncryptionKeypair),
/* harmony export */   getRTCService: () => (/* reexport safe */ _webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.getRTCService),
/* harmony export */   getRemoteService: () => (/* binding */ getRemoteService),
/* harmony export */   getRemoteServiceHTTP: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_5__.getRemoteServiceHTTP),
/* harmony export */   hyphaWebsocketClient: () => (/* binding */ hyphaWebsocketClient),
/* harmony export */   loadRequirements: () => (/* reexport safe */ _utils_index_js__WEBPACK_IMPORTED_MODULE_2__.loadRequirements),
/* harmony export */   login: () => (/* binding */ login),
/* harmony export */   logout: () => (/* binding */ logout),
/* harmony export */   normalizeServerUrlHTTP: () => (/* reexport safe */ _http_client_js__WEBPACK_IMPORTED_MODULE_5__.normalizeServerUrl),
/* harmony export */   publicKeyFromHex: () => (/* reexport safe */ _crypto_js__WEBPACK_IMPORTED_MODULE_1__.publicKeyFromHex),
/* harmony export */   publicKeyToHex: () => (/* reexport safe */ _crypto_js__WEBPACK_IMPORTED_MODULE_1__.publicKeyToHex),
/* harmony export */   registerRTCService: () => (/* reexport safe */ _webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.registerRTCService),
/* harmony export */   schemaFunction: () => (/* reexport safe */ _utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction),
/* harmony export */   setupLocalClient: () => (/* binding */ setupLocalClient)
/* harmony export */ });
/* harmony import */ var _rpc_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./rpc.js */ "./src/rpc.js");
/* harmony import */ var _crypto_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./crypto.js */ "./src/crypto.js");
/* harmony import */ var _utils_index_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./utils/index.js */ "./src/utils/index.js");
/* harmony import */ var _utils_schema_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./utils/schema.js */ "./src/utils/schema.js");
/* harmony import */ var _webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./webrtc-client.js */ "./src/webrtc-client.js");
/* harmony import */ var _http_client_js__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./http-client.js */ "./src/http-client.js");






// Import HTTP client for internal use and re-export


// Re-export HTTP client classes and functions







const MAX_RETRY = 1000000;

class WebsocketRPCConnection {
  constructor(
    server_url,
    client_id,
    workspace,
    token,
    reconnection_token = null,
    timeout = 60,
    WebSocketClass = null,
    token_refresh_interval = 2 * 60 * 60,
    additional_headers = null,
  ) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(server_url && client_id, "server_url and client_id are required");
    this._server_url = server_url;
    this._client_id = client_id;
    this._workspace = workspace;
    this._token = token;
    this._reconnection_token = reconnection_token;
    this._websocket = null;
    this._handle_message = null;
    this._handle_connected = null; // Connection open event handler
    this._handle_disconnected = null; // Disconnection event handler
    this._timeout = timeout;
    this._WebSocketClass = WebSocketClass || WebSocket; // Allow overriding the WebSocket class
    this._closed = false;
    this._legacy_auth = null;
    this.connection_info = null;
    this._enable_reconnect = false;
    this._token_refresh_interval = token_refresh_interval;
    this.manager_id = null;
    this._refresh_token_task = null;
    this._reconnect_timeouts = new Set(); // Track reconnection timeouts
    this._additional_headers = additional_headers;
    this._reconnecting = false; // Mutex to prevent overlapping reconnection attempts
    this._disconnectedNotified = false;
  }

  /**
   * Centralized cleanup method to clear all timers and prevent resource leaks
   */
  _cleanup() {
    // Clear token refresh delay timeout
    if (this._refresh_token_delay) {
      clearTimeout(this._refresh_token_delay);
      this._refresh_token_delay = null;
    }

    // Clear token refresh interval
    if (this._refresh_token_task) {
      clearInterval(this._refresh_token_task);
      this._refresh_token_task = null;
    }

    // Clear all reconnection timeouts
    for (const timeoutId of this._reconnect_timeouts) {
      clearTimeout(timeoutId);
    }
    this._reconnect_timeouts.clear();
  }

  on_message(handler) {
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(handler, "handler is required");
    this._handle_message = handler;
  }

  on_connected(handler) {
    this._handle_connected = handler;
  }

  on_disconnected(handler) {
    this._handle_disconnected = handler;
  }

  async _attempt_connection(server_url, attempt_fallback = true) {
    return new Promise((resolve, reject) => {
      this._legacy_auth = false;
      const websocket = new this._WebSocketClass(server_url);
      websocket.binaryType = "arraybuffer";

      websocket.onopen = () => {
        console.info("WebSocket connection established");
        resolve(websocket);
      };

      websocket.onerror = (event) => {
        console.error("WebSocket connection error:", event);
        reject(new Error(`WebSocket connection error: ${event}`));
      };

      websocket.onclose = (event) => {
        if (event.code === 1003 && attempt_fallback) {
          console.info(
            "Received 1003 error, attempting connection with query parameters.",
          );
          this._legacy_auth = true;
          this._attempt_connection_with_query_params(server_url)
            .then(resolve)
            .catch(reject);
        } else {
          this._notifyDisconnected(event.reason);
        }
      };
    });
  }

  async _attempt_connection_with_query_params(server_url) {
    // Initialize an array to hold parts of the query string
    const queryParamsParts = [];

    // Conditionally add each parameter if it has a non-empty value
    if (this._client_id)
      queryParamsParts.push(`client_id=${encodeURIComponent(this._client_id)}`);
    if (this._workspace)
      queryParamsParts.push(`workspace=${encodeURIComponent(this._workspace)}`);
    if (this._token)
      queryParamsParts.push(`token=${encodeURIComponent(this._token)}`);
    if (this._reconnection_token)
      queryParamsParts.push(
        `reconnection_token=${encodeURIComponent(this._reconnection_token)}`,
      );

    // Join the parts with '&' to form the final query string, prepend '?' if there are any parameters
    const queryString =
      queryParamsParts.length > 0 ? `?${queryParamsParts.join("&")}` : "";

    // Construct the full URL by appending the query string if it exists
    const full_url = server_url + queryString;

    return await this._attempt_connection(full_url, false);
  }

  _establish_connection() {
    return new Promise((resolve, reject) => {
      this._websocket.onmessage = (event) => {
        const data = event.data;
        const first_message = JSON.parse(data);
        if (first_message.type == "connection_info") {
          this.connection_info = first_message;
          if (this._workspace) {
            (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(
              this.connection_info.workspace === this._workspace,
              `Connected to the wrong workspace: ${this.connection_info.workspace}, expected: ${this._workspace}`,
            );
          }
          if (this.connection_info.reconnection_token) {
            this._reconnection_token = this.connection_info.reconnection_token;
          }
          if (this.connection_info.reconnection_token_life_time) {
            // make sure the token refresh interval is less than the token life time
            if (
              this._token_refresh_interval >
              this.connection_info.reconnection_token_life_time / 1.5
            ) {
              console.warn(
                `Token refresh interval is too long (${this._token_refresh_interval}), setting it to 1.5 times of the token life time(${this.connection_info.reconnection_token_life_time}).`,
              );
              this._token_refresh_interval =
                this.connection_info.reconnection_token_life_time / 1.5;
            }
          }
          this.manager_id = this.connection_info.manager_id || null;
          console.log(
            `Successfully connected to the server, workspace: ${this.connection_info.workspace}, manager_id: ${this.manager_id}`,
          );
          if (this.connection_info.announcement) {
            console.log(`${this.connection_info.announcement}`);
          }
          resolve(this.connection_info);
        } else if (first_message.type == "error") {
          const error = "ConnectionAbortedError: " + first_message.message;
          console.error("Failed to connect, " + error);
          reject(new Error(error));
          return;
        } else {
          console.error(
            "ConnectionAbortedError: Unexpected message received from the server:",
            data,
          );
          reject(
            new Error(
              "ConnectionAbortedError: Unexpected message received from the server",
            ),
          );
          return;
        }
      };
    });
  }

  async open() {
    console.log(
      "Creating a new websocket connection to",
      this._server_url.split("?")[0],
    );
    try {
      this._websocket = await this._attempt_connection(this._server_url);
      if (this._legacy_auth) {
        throw new Error(
          "NotImplementedError: Legacy authentication is not supported",
        );
      }
      // Send authentication info as the first message if connected without query params
      const authInfo = JSON.stringify({
        client_id: this._client_id,
        workspace: this._workspace,
        token: this._token,
        reconnection_token: this._reconnection_token,
      });
      this._websocket.send(authInfo);
      // Wait for the first message from the server
      await (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.waitFor)(
        this._establish_connection(),
        this._timeout,
        "Failed to receive the first message from the server",
      );
      if (this._token_refresh_interval > 0) {
        this._refresh_token_delay = setTimeout(() => {
          this._refresh_token_delay = null;
          if (this._closed) return;
          this._send_refresh_token();
          this._refresh_token_task = setInterval(() => {
            this._send_refresh_token();
          }, this._token_refresh_interval * 1000);
        }, 2000);
      }
      // Listen to messages from the server
      this._enable_reconnect = true;
      this._closed = false;
      this._disconnectedNotified = false;
      this._websocket.onmessage = (event) => {
        if (typeof event.data === "string") {
          const parsedData = JSON.parse(event.data);
          // Check if the message is a reconnection token
          if (parsedData.type === "reconnection_token") {
            this._reconnection_token = parsedData.reconnection_token;
            // console.log("Reconnection token received");
          } else {
            console.log("Received message from the server:", parsedData);
          }
        } else {
          this._handle_message(event.data);
        }
      };

      this._websocket.onerror = (event) => {
        console.error("WebSocket connection error:", event);
        // Clean up timers on error
        this._cleanup();
      };

      this._websocket.onclose = this._handle_close.bind(this);

      if (this._handle_connected) {
        this._handle_connected(this.connection_info);
      }
      return this.connection_info;
    } catch (error) {
      // Clean up any timers that might have been set up before the error
      this._cleanup();
      console.error(
        "Failed to connect to",
        this._server_url.split("?")[0],
        error,
      );
      throw error;
    }
  }

  _send_refresh_token() {
    if (this._websocket && this._websocket.readyState === WebSocket.OPEN) {
      const refreshMessage = JSON.stringify({ type: "refresh_token" });
      this._websocket.send(refreshMessage);
      // console.log("Requested refresh token");
    }
  }

  _notifyDisconnected(reason) {
    if (this._disconnectedNotified) return;
    this._disconnectedNotified = true;
    if (this._handle_disconnected) {
      this._handle_disconnected(reason);
    }
  }

  _handle_close(event) {
    if (
      !this._closed &&
      this._websocket &&
      this._websocket.readyState === WebSocket.CLOSED
    ) {
      // Clean up timers when connection closes
      this._cleanup();
      // Reset the guard so reconnection can re-notify on next disconnect
      this._disconnectedNotified = false;

      // Even if it's a graceful closure (codes 1000, 1001), if it wasn't user-initiated,
      // we should attempt to reconnect (e.g., server restart, k8s upgrade)
      if (this._enable_reconnect) {
        if ([1000, 1001].includes(event.code)) {
          console.warn(
            `Websocket connection closed gracefully by server (code: ${event.code}): ${event.reason} - attempting reconnect`,
          );
        } else {
          console.warn(
            "Websocket connection closed unexpectedly (code: %s): %s",
            event.code,
            event.reason,
          );
        }

        // Notify the RPC layer immediately so it can reject pending calls
        this._notifyDisconnected(event.reason);

        // Prevent overlapping reconnection attempts
        if (this._reconnecting) {
          console.debug("Reconnection already in progress, skipping");
          return;
        }
        this._reconnecting = true;

        let retry = 0;
        const baseDelay = 1000; // Start with 1 second
        const maxDelay = 60000; // Maximum delay of 60 seconds
        const maxJitter = 0.1; // Maximum jitter factor

        const reconnect = async () => {
          // Check if we were explicitly closed
          if (this._closed) {
            console.info("Connection was closed, stopping reconnection");
            this._reconnecting = false;
            return;
          }

          try {
            console.warn(
              `Reconnecting to ${this._server_url.split("?")[0]} (attempt #${retry})`,
            );
            // Open the connection, this will trigger the on_connected callback
            await this.open();

            // Wait a short time for services to be registered
            // This gives time for the on_connected callback to complete
            // which includes re-registering all services to the server
            await new Promise((resolve) => setTimeout(resolve, 500));

            console.warn(
              `Successfully reconnected to server ${this._server_url} (services re-registered)`,
            );
            this._reconnecting = false;
          } catch (e) {
            if (`${e}`.includes("ConnectionAbortedError:")) {
              console.warn("Server refused to reconnect:", e);
              this._closed = true;
              this._reconnecting = false;
              this._notifyDisconnected(`Server refused reconnection: ${e}`);
              return;
            } else if (`${e}`.includes("NotImplementedError:")) {
              console.error(
                `${e}\nIt appears that you are trying to connect to a hypha server that is older than 0.20.0, please upgrade the hypha server or use the websocket client in imjoy-rpc(https://www.npmjs.com/package/imjoy-rpc) instead`,
              );
              this._closed = true;
              this._reconnecting = false;
              this._notifyDisconnected(`Server too old: ${e}`);
              return;
            }

            // Log specific error types for better debugging
            if (e.name === "NetworkError" || e.message.includes("network")) {
              console.error(`Network error during reconnection: ${e.message}`);
            } else if (
              e.name === "TimeoutError" ||
              e.message.includes("timeout")
            ) {
              console.error(
                `Connection timeout during reconnection: ${e.message}`,
              );
            } else {
              console.error(
                `Unexpected error during reconnection: ${e.message}`,
              );
            }

            // Calculate exponential backoff with jitter
            const delay = Math.min(baseDelay * Math.pow(2, retry), maxDelay);
            // Add jitter to prevent thundering herd
            const jitter = (Math.random() * 2 - 1) * maxJitter * delay;
            const finalDelay = Math.max(100, delay + jitter);

            console.debug(
              `Waiting ${(finalDelay / 1000).toFixed(2)}s before next reconnection attempt`,
            );

            // Track the reconnection timeout to prevent leaks
            const timeoutId = setTimeout(async () => {
              this._reconnect_timeouts.delete(timeoutId);

              // Check if connection was restored externally
              if (
                this._websocket &&
                this._websocket.readyState === WebSocket.OPEN
              ) {
                console.info("Connection restored externally");
                this._reconnecting = false;
                return;
              }

              // Check if we were explicitly closed
              if (this._closed) {
                console.info("Connection was closed, stopping reconnection");
                this._reconnecting = false;
                return;
              }

              retry += 1;
              if (retry < MAX_RETRY) {
                await reconnect();
              } else {
                console.error(
                  `Failed to reconnect after ${MAX_RETRY} attempts, giving up.`,
                );
                this._closed = true;
                this._reconnecting = false;
                this._notifyDisconnected("Max reconnection attempts exceeded");
              }
            }, finalDelay);
            this._reconnect_timeouts.add(timeoutId);
          }
        };
        reconnect();
      }
    } else {
      // Clean up timers in all cases
      this._cleanup();
      this._notifyDisconnected(event.reason);
    }
  }

  async emit_message(data) {
    if (this._closed) {
      throw new Error("Connection is closed");
    }
    if (!this._websocket || this._websocket.readyState !== WebSocket.OPEN) {
      await this.open();
    }
    try {
      this._websocket.send(data);
    } catch (exp) {
      console.error(`Failed to send data, error: ${exp}`);
      throw exp;
    }
  }

  disconnect(reason) {
    this._closed = true;
    this._reconnecting = false;
    // Ensure websocket is closed if it exists and is not already closed or closing
    if (
      this._websocket &&
      this._websocket.readyState !== WebSocket.CLOSED &&
      this._websocket.readyState !== WebSocket.CLOSING
    ) {
      this._websocket.close(1000, reason);
    }
    // Use centralized cleanup to clear all timers
    this._cleanup();
    console.info(`WebSocket connection disconnected (${reason})`);
  }
}

function normalizeServerUrl(server_url) {
  if (!server_url) throw new Error("server_url is required");
  if (server_url.startsWith("http://")) {
    server_url =
      server_url.replace("http://", "ws://").replace(/\/$/, "") + "/ws";
  } else if (server_url.startsWith("https://")) {
    server_url =
      server_url.replace("https://", "wss://").replace(/\/$/, "") + "/ws";
  }
  return server_url;
}

/**
 * Login to the hypha server.
 *
 * Configuration options:
 *   server_url: The server URL (required)
 *   workspace: Target workspace (optional)
 *   login_service_id: Login service ID (default: "public/hypha-login")
 *   expires_in: Token expiration time (optional)
 *   login_timeout: Timeout for login process (default: 60)
 *   login_callback: Callback function for login URL (optional)
 *   profile: Whether to return user profile (optional)
 *   additional_headers: Additional HTTP headers (optional)
 *   transport: Transport type - "websocket" (default) or "http"
 */
async function login(config) {
  const service_id = config.login_service_id || "public/hypha-login";
  const workspace = config.workspace;
  const expires_in = config.expires_in;
  const timeout = config.login_timeout || 60;
  const callback = config.login_callback;
  const profile = config.profile;
  const additional_headers = config.additional_headers;
  const transport = config.transport || "websocket";

  const server = await connectToServer({
    name: "initial login client",
    server_url: config.server_url,
    additional_headers: additional_headers,
    transport: transport,
  });
  try {
    const svc = await server.getService(service_id);
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(svc, `Failed to get the login service: ${service_id}`);
    let context;
    if (workspace) {
      context = await svc.start({ workspace, expires_in, _rkwargs: true });
    } else {
      context = await svc.start();
    }
    if (callback) {
      await callback(context);
    } else {
      console.log(`Please open your browser and login at ${context.login_url}`);
    }
    return await svc.check(context.key, { timeout, profile, _rkwargs: true });
  } catch (error) {
    throw error;
  } finally {
    await server.disconnect();
  }
}

/**
 * Logout from the hypha server.
 *
 * Configuration options:
 *   server_url: The server URL (required)
 *   login_service_id: Login service ID (default: "public/hypha-login")
 *   logout_callback: Callback function for logout URL (optional)
 *   additional_headers: Additional HTTP headers (optional)
 *   transport: Transport type - "websocket" (default) or "http"
 */
async function logout(config) {
  const service_id = config.login_service_id || "public/hypha-login";
  const callback = config.logout_callback;
  const additional_headers = config.additional_headers;
  const transport = config.transport || "websocket";

  const server = await connectToServer({
    name: "initial logout client",
    server_url: config.server_url,
    additional_headers: additional_headers,
    transport: transport,
  });
  try {
    const svc = await server.getService(service_id);
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(svc, `Failed to get the login service: ${service_id}`);

    // Check if logout function exists for backward compatibility
    if (!svc.logout) {
      throw new Error(
        "Logout is not supported by this server. " +
          "Please upgrade the Hypha server to a version that supports logout.",
      );
    }

    const context = await svc.logout({});
    if (callback) {
      await callback(context);
    } else {
      console.log(
        `Please open your browser to logout at ${context.logout_url}`,
      );
    }
    return context;
  } catch (error) {
    throw error;
  } finally {
    await server.disconnect();
  }
}

async function webrtcGetService(wm, query, config) {
  config = config || {};
  // Default to "auto" since this wrapper is only used when connection was
  // established with webrtc: true
  const webrtc = config.webrtc !== undefined ? config.webrtc : "auto";
  const webrtc_config = config.webrtc_config;
  const encryptionPublicKey = config.encryption_public_key;
  if (config.webrtc !== undefined) delete config.webrtc;
  if (config.webrtc_config !== undefined) delete config.webrtc_config;
  if (config.encryption_public_key !== undefined)
    delete config.encryption_public_key;
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(
    [undefined, true, false, "auto"].includes(webrtc),
    "webrtc must be true, false or 'auto'",
  );

  const svc = await wm.getService(query, config);
  if (encryptionPublicKey) {
    (0,_rpc_js__WEBPACK_IMPORTED_MODULE_0__._applyEncryptionKeyToService)(svc, (0,_crypto_js__WEBPACK_IMPORTED_MODULE_1__.publicKeyFromHex)(encryptionPublicKey));
  }
  if (webrtc === true || webrtc === "auto") {
    if (svc.id.includes(":") && svc.id.includes("/")) {
      try {
        // Extract remote client_id from service id
        // svc.id format: "workspace/client_id:service_id"
        const wsAndClient = svc.id.split(":")[0]; // "workspace/client_id"
        const parts = wsAndClient.split("/");
        const remoteClientId = parts[parts.length - 1]; // "client_id"
        const remoteWorkspace = parts.slice(0, -1).join("/"); // "workspace"
        const remoteRtcServiceId = `${remoteWorkspace}/${remoteClientId}-rtc`;
        const peer = await (0,_webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.getRTCService)(wm, remoteRtcServiceId, webrtc_config);
        const rtcSvc = await peer.getService(svc.id.split(":")[1], config);
        rtcSvc._webrtc = true;
        rtcSvc._peer = peer;
        rtcSvc._service = svc;
        return rtcSvc;
      } catch (e) {
        console.warn(
          "Failed to get webrtc service, using websocket connection",
          e,
        );
      }
    }
    if (webrtc === true) {
      throw new Error("Failed to get the service via webrtc");
    }
  }
  return svc;
}

async function connectToServer(config) {
  // Support HTTP transport via transport option
  const transport = config.transport || "websocket";
  if (transport === "http") {
    return await (0,_http_client_js__WEBPACK_IMPORTED_MODULE_5__.connectToServerHTTP)(config);
  }

  if (config.server) {
    config.server_url = config.server_url || config.server.url;
    config.WebSocketClass =
      config.WebSocketClass || config.server.WebSocketClass;
  }
  let clientId = config.client_id;
  if (!clientId) {
    clientId = (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.randId)();
    config.client_id = clientId;
  }
  if (Object.keys(config).length === 0) {
    if (typeof process !== "undefined" && process.env) {
      // Node.js
      config.server_url = process.env.HYPHA_SERVER_URL;
      config.token = process.env.HYPHA_TOKEN;
      config.client_id = process.env.HYPHA_CLIENT_ID;
      config.workspace = process.env.HYPHA_WORKSPACE;
    } else if (typeof self !== "undefined" && self.env) {
      // WebWorker (only if you inject self.env manually)
      config.server_url = self.env.HYPHA_SERVER_URL;
      config.token = self.env.HYPHA_TOKEN;
      config.client_id = self.env.HYPHA_CLIENT_ID;
      config.workspace = self.env.HYPHA_WORKSPACE;
    } else if (typeof globalThis !== "undefined" && globalThis.env) {
      // Browser (only if you define globalThis.env beforehand)
      config.server_url = globalThis.env.HYPHA_SERVER_URL;
      config.token = globalThis.env.HYPHA_TOKEN;
      config.client_id = globalThis.env.HYPHA_CLIENT_ID;
      config.workspace = globalThis.env.HYPHA_WORKSPACE;
    }
  }

  let server_url = normalizeServerUrl(config.server_url);

  let connection = new WebsocketRPCConnection(
    server_url,
    clientId,
    config.workspace,
    config.token,
    config.reconnection_token,
    config.method_timeout || 60,
    config.WebSocketClass,
    config.token_refresh_interval,
    config.additional_headers,
  );
  const connection_info = await connection.open();
  (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(
    connection_info,
    "Failed to connect to the server, no connection info obtained. This issue is most likely due to an outdated Hypha server version. Please use `imjoy-rpc` for compatibility, or upgrade the Hypha server to the latest version.",
  );
  // wait for 0.5 seconds
  await new Promise((resolve) => setTimeout(resolve, 100));
  // Ensure manager_id is set before proceeding
  if (!connection.manager_id) {
    console.warn("Manager ID not set immediately, waiting...");

    // Wait for manager_id to be set with timeout
    const maxWaitTime = 5000; // 5 seconds
    const checkInterval = 100; // 100ms
    const startTime = Date.now();

    while (!connection.manager_id && Date.now() - startTime < maxWaitTime) {
      await new Promise((resolve) => setTimeout(resolve, checkInterval));
    }

    if (!connection.manager_id) {
      console.error("Manager ID still not set after waiting");
      throw new Error("Failed to get manager ID from server");
    } else {
      console.info(`Manager ID set after waiting: ${connection.manager_id}`);
    }
  }
  if (config.workspace && connection_info.workspace !== config.workspace) {
    throw new Error(
      `Connected to the wrong workspace: ${connection_info.workspace}, expected: ${config.workspace}`,
    );
  }

  const workspace = connection_info.workspace;
  const rpc = new _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC(connection, {
    client_id: clientId,
    workspace,
    default_context: { connection_type: "websocket" },
    name: config.name,
    method_timeout: config.method_timeout,
    app_id: config.app_id,
    server_base_url: connection_info.public_base_url,
    long_message_chunk_size: config.long_message_chunk_size,
    encryption: config.encryption || false,
    encryption_private_key: config.encryption_private_key || null,
    encryption_public_key: config.encryption_public_key || null,
  });
  await rpc.waitFor("services_registered", config.method_timeout || 120);
  const wm = await rpc.get_manager_service({
    timeout: config.method_timeout,
    case_conversion: "camel",
    kwargs_expansion: config.kwargs_expansion || false,
  });
  wm.rpc = rpc;

  async function _export(api) {
    api.id = "default";
    api.name = api.name || config.name || api.id;
    api.description = api.description || config.description;
    await rpc.register_service(api, { overwrite: true });
  }

  async function getApp(clientId) {
    clientId = clientId || "*";
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(!clientId.includes(":"), "clientId should not contain ':'");
    if (!clientId.includes("/")) {
      clientId = connection_info.workspace + "/" + clientId;
    }
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(
      clientId.split("/").length === 2,
      "clientId should match pattern workspace/clientId",
    );
    return await wm.getService(`${clientId}:default`);
  }

  async function listApps(ws) {
    ws = ws || workspace;
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(!ws.includes(":"), "workspace should not contain ':'");
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.assert)(!ws.includes("/"), "workspace should not contain '/'");
    const query = { workspace: ws, service_id: "default" };
    return await wm.listServices(query);
  }

  if (connection_info) {
    wm.config = Object.assign(wm.config, connection_info);
  }
  wm.export = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(_export, {
    name: "export",
    description: "Export the api.",
    parameters: {
      properties: { api: { description: "The api to export", type: "object" } },
      required: ["api"],
      type: "object",
    },
  });
  wm.getApp = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(getApp, {
    name: "getApp",
    description: "Get the app.",
    parameters: {
      properties: {
        clientId: { default: "*", description: "The clientId", type: "string" },
      },
      type: "object",
    },
  });
  wm.listApps = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(listApps, {
    name: "listApps",
    description: "List the apps.",
    parameters: {
      properties: {
        workspace: {
          default: workspace,
          description: "The workspace",
          type: "string",
        },
      },
      type: "object",
    },
  });
  wm.disconnect = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.disconnect.bind(rpc), {
    name: "disconnect",
    description: "Disconnect from the server.",
    parameters: { type: "object", properties: {}, required: [] },
  });
  wm.registerCodec = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.register_codec.bind(rpc), {
    name: "registerCodec",
    description: "Register a codec for the webrtc connection",
    parameters: {
      type: "object",
      properties: {
        codec: {
          type: "object",
          description: "Codec to register",
          properties: {
            name: { type: "string" },
            type: {},
            encoder: { type: "function" },
            decoder: { type: "function" },
          },
        },
      },
    },
  });

  wm.emit = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.emit.bind(rpc), {
    name: "emit",
    description: "Emit a message.",
    parameters: {
      properties: { data: { description: "The data to emit", type: "object" } },
      required: ["data"],
      type: "object",
    },
  });

  wm.on = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.on.bind(rpc), {
    name: "on",
    description: "Register a message handler.",
    parameters: {
      properties: {
        event: { description: "The event to listen to", type: "string" },
        handler: { description: "The handler function", type: "function" },
      },
      required: ["event", "handler"],
      type: "object",
    },
  });

  wm.off = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.off.bind(rpc), {
    name: "off",
    description: "Remove a message handler.",
    parameters: {
      properties: {
        event: { description: "The event to remove", type: "string" },
        handler: { description: "The handler function", type: "function" },
      },
      required: ["event", "handler"],
      type: "object",
    },
  });

  wm.once = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.once.bind(rpc), {
    name: "once",
    description: "Register a one-time message handler.",
    parameters: {
      properties: {
        event: { description: "The event to listen to", type: "string" },
        handler: { description: "The handler function", type: "function" },
      },
      required: ["event", "handler"],
      type: "object",
    },
  });

  wm.getServiceSchema = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.get_service_schema, {
    name: "getServiceSchema",
    description: "Get the service schema.",
    parameters: {
      properties: {
        service: {
          description: "The service to extract schema",
          type: "object",
        },
      },
      required: ["service"],
      type: "object",
    },
  });

  wm.registerService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.register_service.bind(rpc), {
    name: "registerService",
    description: "Register a service.",
    parameters: {
      properties: {
        service: { description: "The service to register", type: "object" },
        force: {
          default: false,
          description: "Force to register the service",
          type: "boolean",
        },
      },
      required: ["service"],
      type: "object",
    },
  });
  wm.unregisterService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(rpc.unregister_service.bind(rpc), {
    name: "unregisterService",
    description: "Unregister a service.",
    parameters: {
      properties: {
        service: {
          description: "The service id to unregister",
          type: "string",
        },
        notify: {
          default: true,
          description: "Notify the workspace manager",
          type: "boolean",
        },
      },
      required: ["service"],
      type: "object",
    },
  });
  if (connection.manager_id) {
    rpc.on("force-exit", async (message) => {
      if (message.from === "*/" + connection.manager_id) {
        console.log("Disconnecting from server, reason:", message.reason);
        await rpc.disconnect();
      }
    });
  }
  if (config.webrtc) {
    await (0,_webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.registerRTCService)(wm, `${clientId}-rtc`, config.webrtc_config);
    // make a copy of wm, so webrtc can use the original wm.getService
    const _wm = Object.assign({}, wm);
    const description = _wm.getService.__schema__.description;
    // TODO: Fix the schema for adding options for webrtc
    const parameters = _wm.getService.__schema__.parameters;
    wm.getService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(webrtcGetService.bind(null, _wm), {
      name: "getService",
      description,
      parameters,
    });

    wm.getRTCService = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(_webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.getRTCService.bind(null, wm), {
      name: "getRTCService",
      description: "Get the webrtc connection, returns a peer connection.",
      parameters: {
        properties: {
          config: {
            description: "The config for the webrtc service",
            type: "object",
          },
        },
        required: ["config"],
        type: "object",
      },
    });
  } else {
    const _getService = wm.getService;
    wm.getService = async (query, config) => {
      config = config || {};
      const encryptionPublicKey = config.encryption_public_key;
      delete config.encryption_public_key;
      const svc = await _getService(query, config);
      if (encryptionPublicKey) {
        (0,_rpc_js__WEBPACK_IMPORTED_MODULE_0__._applyEncryptionKeyToService)(svc, (0,_crypto_js__WEBPACK_IMPORTED_MODULE_1__.publicKeyFromHex)(encryptionPublicKey));
      }
      return svc;
    };
    wm.getService.__schema__ = _getService.__schema__;
  }

  async function registerProbes(probes) {
    probes.id = "probes";
    probes.name = "Probes";
    probes.config = { visibility: "public" };
    probes.type = "probes";
    probes.description = `Probes Service, visit ${server_url}/${workspace}services/probes for the available probes.`;
    return await wm.registerService(probes, { overwrite: true });
  }

  wm.registerProbes = (0,_utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction)(registerProbes, {
    name: "registerProbes",
    description: "Register probes service",
    parameters: {
      properties: {
        probes: {
          description:
            "The probes to register, e.g. {'liveness': {'type': 'function', 'description': 'Check the liveness of the service'}}",
          type: "object",
        },
      },
      required: ["probes"],
      type: "object",
    },
  });
  return wm;
}

async function getRemoteService(serviceUri, config = {}) {
  const { serverUrl, workspace, clientId, serviceId, appId } =
    (0,_utils_index_js__WEBPACK_IMPORTED_MODULE_2__.parseServiceUrl)(serviceUri);
  const fullServiceId = `${workspace}/${clientId}:${serviceId}@${appId}`;

  if (config.serverUrl) {
    if (config.serverUrl !== serverUrl) {
      throw new Error(
        "server_url in config does not match the server_url in the url",
      );
    }
  }
  config.serverUrl = serverUrl;
  const server = await connectToServer(config);
  return await server.getService(fullServiceId);
}

class LocalWebSocket {
  constructor(url, client_id, workspace) {
    this.url = url;
    this.onopen = () => {};
    this.onmessage = () => {};
    this.onclose = () => {};
    this.onerror = () => {};
    this.client_id = client_id;
    this.workspace = workspace;
    const context = typeof window !== "undefined" ? window : self;
    const isWindow = typeof window !== "undefined";
    this.postMessage = (message) => {
      if (isWindow) {
        window.parent.postMessage(message, "*");
      } else {
        self.postMessage(message);
      }
    };

    this.readyState = WebSocket.CONNECTING;
    this._context = context;
    this._messageListener = (event) => {
      const { type, data, to } = event.data;
      if (to !== this.client_id) {
        return;
      }
      switch (type) {
        case "message":
          if (this.readyState === WebSocket.OPEN && this.onmessage) {
            this.onmessage({ data: data });
          }
          break;
        case "connected":
          this.readyState = WebSocket.OPEN;
          this.onopen(event);
          break;
        case "closed":
          this.readyState = WebSocket.CLOSED;
          this.onclose(event);
          break;
        default:
          break;
      }
    };
    context.addEventListener("message", this._messageListener, false);

    if (!this.client_id) throw new Error("client_id is required");
    if (!this.workspace) throw new Error("workspace is required");
    this.postMessage({
      type: "connect",
      url: this.url,
      from: this.client_id,
      workspace: this.workspace,
    });
  }

  send(data) {
    if (this.readyState === WebSocket.OPEN) {
      this.postMessage({
        type: "message",
        data: data,
        from: this.client_id,
        workspace: this.workspace,
      });
    }
  }

  close() {
    this.readyState = WebSocket.CLOSING;
    this.postMessage({
      type: "close",
      from: this.client_id,
      workspace: this.workspace,
    });
    if (this._context && this._messageListener) {
      this._context.removeEventListener(
        "message",
        this._messageListener,
        false,
      );
      this._messageListener = null;
    }
    this.onclose();
  }

  addEventListener(type, listener) {
    if (type === "message") {
      this.onmessage = listener;
    }
    if (type === "open") {
      this.onopen = listener;
    }
    if (type === "close") {
      this.onclose = listener;
    }
    if (type === "error") {
      this.onerror = listener;
    }
  }
}

// Also export a hyphaWebsocketClient namespace for backwards compatibility.
// hypha-core's deno build does: import { hyphaWebsocketClient } from 'hypha-rpc'
// The UMD build wraps everything under this name via webpack's `library` option,
// but the ESM build exports flat, so we need this explicit re-export.
const hyphaWebsocketClient = {
  RPC: _rpc_js__WEBPACK_IMPORTED_MODULE_0__.RPC,
  API_VERSION: _rpc_js__WEBPACK_IMPORTED_MODULE_0__.API_VERSION,
  schemaFunction: _utils_schema_js__WEBPACK_IMPORTED_MODULE_3__.schemaFunction,
  loadRequirements: _utils_index_js__WEBPACK_IMPORTED_MODULE_2__.loadRequirements,
  login,
  logout,
  connectToServer,
  connectToServerHTTP: _http_client_js__WEBPACK_IMPORTED_MODULE_5__.connectToServerHTTP,
  getRemoteService,
  getRemoteServiceHTTP: _http_client_js__WEBPACK_IMPORTED_MODULE_5__.getRemoteServiceHTTP,
  getRTCService: _webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.getRTCService,
  registerRTCService: _webrtc_client_js__WEBPACK_IMPORTED_MODULE_4__.registerRTCService,
  get setupLocalClient() { return setupLocalClient; },
  get LocalWebSocket() { return LocalWebSocket; },
  HTTPStreamingRPCConnection: _http_client_js__WEBPACK_IMPORTED_MODULE_5__.HTTPStreamingRPCConnection,
  normalizeServerUrlHTTP: _http_client_js__WEBPACK_IMPORTED_MODULE_5__.normalizeServerUrl,
};

function setupLocalClient({
  enable_execution = false,
  on_ready = null,
}) {
  return new Promise((resolve, reject) => {
    const context = typeof window !== "undefined" ? window : self;
    const isWindow = typeof window !== "undefined";
    context.addEventListener(
      "message",
      (event) => {
        const {
          type,
          server_url,
          workspace,
          client_id,
          token,
          method_timeout,
          name,
          config,
        } = event.data;

        if (type === "initializeHyphaClient") {
          if (!server_url || !workspace || !client_id) {
            console.error("server_url, workspace, and client_id are required.");
            return;
          }

          if (!server_url.startsWith("https://local-hypha-server:")) {
            console.error(
              "server_url should start with https://local-hypha-server:",
            );
            return;
          }

          class FixedLocalWebSocket extends LocalWebSocket {
            constructor(url) {
              // Call the parent class's constructor with fixed values
              super(url, client_id, workspace);
            }
          }
          connectToServer({
            server_url,
            workspace,
            client_id,
            token,
            method_timeout,
            name,
            WebSocketClass: FixedLocalWebSocket,
          }).then(async (server) => {
            globalThis.api = server;
            try {
              // for iframe
              if (isWindow && enable_execution) {
                function loadScript(script) {
                  return new Promise((resolve, reject) => {
                    const scriptElement = document.createElement("script");
                    scriptElement.innerHTML = script.content;
                    scriptElement.lang = script.lang;

                    scriptElement.onload = () => resolve();
                    scriptElement.onerror = (e) => reject(e);

                    document.head.appendChild(scriptElement);
                  });
                }
                if (config.styles && config.styles.length > 0) {
                  for (const style of config.styles) {
                    const styleElement = document.createElement("style");
                    styleElement.innerHTML = style.content;
                    styleElement.lang = style.lang;
                    document.head.appendChild(styleElement);
                  }
                }
                if (config.links && config.links.length > 0) {
                  for (const link of config.links) {
                    const linkElement = document.createElement("a");
                    linkElement.href = link.url;
                    linkElement.innerText = link.text;
                    document.body.appendChild(linkElement);
                  }
                }
                if (config.windows && config.windows.length > 0) {
                  for (const w of config.windows) {
                    document.body.innerHTML = w.content;
                    break;
                  }
                }
                if (config.scripts && config.scripts.length > 0) {
                  for (const script of config.scripts) {
                    if (script.lang !== "javascript")
                      throw new Error("Only javascript scripts are supported");
                    await loadScript(script); // Await the loading of each script
                  }
                }
              }
              // for web worker
              else if (
                !isWindow &&
                enable_execution &&
                config.scripts &&
                config.scripts.length > 0
              ) {
                for (const script of config.scripts) {
                  if (script.lang !== "javascript")
                    throw new Error("Only javascript scripts are supported");
                  eval(script.content);
                }
              }

              if (on_ready) {
                await on_ready(server, config);
              }
              resolve(server);
            } catch (e) {
              reject(e);
            }
          });
        }
      },
      false,
    );
    if (isWindow) {
      window.parent.postMessage({ type: "hyphaClientReady" }, "*");
    } else {
      self.postMessage({ type: "hyphaClientReady" });
    }
  });
}

})();

/******/ 	return __webpack_exports__;
/******/ })()
;
});
//# sourceMappingURL=hypha-rpc-websocket.js.map