/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;

struct integrate_functor {
  float deltaTime;

  __host__ __device__ integrate_functor(float delta_time)
      : deltaTime(delta_time) {}

  template <typename Tuple>
  __device__ void operator()(Tuple t) {
    volatile float4 posData = thrust::get<0>(t);
    volatile float4 velData = thrust::get<1>(t);
    float3 pos = make_float3(posData.x, posData.y, posData.z);
    float3 vel = make_float3(velData.x, velData.y, velData.z);

    vel += params.gravity * deltaTime;
    vel *= params.globalDamping;

    // new position = old position + velocity * deltaTime
    pos += vel * deltaTime;

// set this to zero to disable collisions with cube sides
#if 1

    float x_bound = 0.9f;
    float y_bound = x_bound;
    float boundaryForce = 0.001f;
    if (pos.x > 1.0f * x_bound - params.particleRadius) {
      //pos.x = 1.0f * x_bound - params.particleRadius;
      //vel.x *= params.boundaryDamping;
      vel.x -= boundaryForce;
    }

    if (pos.x < -1.0f * x_bound + params.particleRadius) {
      //pos.x = -1.0f * x_bound + params.particleRadius;
      //vel.x *= params.boundaryDamping;
        vel.x += boundaryForce;
    }

    if (pos.y > 1.0f * y_bound - params.particleRadius) {
      //pos.y = 1.0f - params.particleRadius;
      //vel.y *= params.boundaryDamping;
      vel.y -= boundaryForce;
    }

    //if (pos.z > 1.0f - params.particleRadius) {
    //  pos.z = 1.0f - params.particleRadius;
    //  vel.z *= params.boundaryDamping;
    //}

    //if (pos.z < -1.0f + params.particleRadius) {
    //  pos.z = -1.0f + params.particleRadius;
    //  vel.z *= params.boundaryDamping;
    //}
    if (pos.y < -1.0f * y_bound + params.particleRadius) {
        //pos.y = -1.0f + params.particleRadius;
        //vel.y *= params.boundaryDamping;
        vel.y += boundaryForce;
    }
#endif

    

    // store new position and velocity
    thrust::get<0>(t) = make_float4(pos, posData.w);
    thrust::get<1>(t) = make_float4(vel, velData.w);
  }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p) {
  int3 gridPos;
  gridPos.x = floorf((p.x - params.worldOrigin.x) / params.cellSize.x);
  gridPos.y = floorf((p.y - params.worldOrigin.y) / params.cellSize.y);
  gridPos.z = 0;                                 // floorf((p.z - params.worldOrigin.z) / params.cellSize.z);
  return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos) {
  gridPos.x = gridPos.x &
              (params.gridSize.x - 1);  // wrap grid, assumes size is power of 2
  gridPos.y = gridPos.y & (params.gridSize.y - 1);
  gridPos.z = gridPos.z & (params.gridSize.z - 1);
  return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) +
         __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__ void calcHashD(uint *gridParticleHash,   // output
                          uint *gridParticleIndex,  // output
                          float4 *pos,              // input: positions
                          uint numParticles) {
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (index >= numParticles) return;

  volatile float4 p = pos[index];

  // get address in grid
  int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
  uint hash = calcGridHash(gridPos);

  // store grid hash and particle index
  gridParticleHash[index] = hash;
  gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderDataAndFindCellStartD(
    uint *cellStart,          // output: cell start index
    uint *cellEnd,            // output: cell end index
    float4 *sortedPos,        // output: sorted positions
    float4 *sortedVel,        // output: sorted velocities
    uint *gridParticleHash,   // input: sorted grid hashes
    uint *gridParticleIndex,  // input: sorted particle indices
    float4 *oldPos,           // input: sorted position array
    float4 *oldVel,           // input: sorted velocity array
    uint numParticles) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ uint sharedHash[];  // blockSize + 1 elements
  uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  uint hash;

  // handle case when no. of particles not multiple of block size
  if (index < numParticles) {
    hash = gridParticleHash[index];

    // Load hash data into shared memory so that we can look
    // at neighboring particle's hash value without loading
    // two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;

    if (index > 0 && threadIdx.x == 0) {
      // first thread in block must load neighbor particle hash
      sharedHash[0] = gridParticleHash[index - 1];
    }
  }

  cg::sync(cta);

  if (index < numParticles) {
    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    if (index == 0 || hash != sharedHash[threadIdx.x]) {
      cellStart[hash] = index;

      if (index > 0) cellEnd[sharedHash[threadIdx.x]] = index;
    }

    if (index == numParticles - 1) {
      cellEnd[hash] = index + 1;
    }

    // Now use the sorted index to reorder the pos and vel data
    uint sortedIndex = gridParticleIndex[index];
    float4 pos = oldPos[sortedIndex];
    float4 vel = oldVel[sortedIndex];

    sortedPos[index] = pos;
    sortedVel[index] = vel;
  }
}

//// collide two spheres using DEM method
//__device__ float3 collideSpheres_Hard(float3 posA, float3 posB, float3 velA,
//                                 float3 velB, float radiusA, float radiusB,
//                                 float attraction) {
//  
//
//  // calculate relative position
//  float3 relPos = posB - posA;
//
//  float dist = length(relPos);
//  float collideDist = radiusA + radiusB;
//
//  float3 force = make_float3(0.0f);
//
//  if (dist < collideDist) {
//    float3 norm = relPos / dist;
//
//    // relative velocity
//    float3 relVel = velB - velA;
//
//    // relative tangential velocity
//    float3 tanVel = relVel - (dot(relVel, norm) * norm);
//
//    // spring force
//    force = -params.spring * (collideDist - dist) * norm;
//    // dashpot (damping) force
//    force += params.damping * relVel;
//    // tangential shear force
//    force += params.shear * tanVel;
//    // attraction
//    force += attraction * relPos;
//  }
//
//  return force;
//}

__device__ float forceMag_lj(float dist)
{
    float epsilon = 0.00000001f;
    dist += epsilon;
    float dist2 = dist * dist;
    float dist4 = dist2 * dist2;
    float dist8 = dist4 * dist4;

    return 1.0f / (dist8 * dist4) - 1.0f / (dist4 * dist2);
}


//https://www.desmos.com/calculator/vjgbhagq9c
//https://www.desmos.com/calculator/ziurgjgunh
__device__ float forceMag_lj_from_potential(float dist) {
    /*float a = 0.1;
    float b = 0.79;
    float c = 4;
    float d = 4;
    float f = 1.2;
    float g = 3.7;
    float cf = c * f;
    float dg = d * g;*/
    if (dist < 1.0) {
        return 50.0 * dist - 50;
    }
    const float a = 0.4;
    const float b = 0.6;
    const float c = 8;
    const float d = 8;
    const float f = 8.3;
    const float g = 10.8;
    const float cf = c * f;
    const float dg = d * g;

    float dist2 = dist * dist;
    float dist4 = dist2 * dist2;
    float dist8 = dist4 * dist4;
    float a_dist8 = dist8 + a;
    float a_dist8_2 = a_dist8 * a_dist8;
    float b_dist8 = dist8 + b;
    float b_dist8_2 = b_dist8 * b_dist8;


    return ( (dg * dist8 / b_dist8_2) - (cf * dist8 / a_dist8_2) ) / dist;
}


__device__ float forceMag_custom_peak(float dist) {
    float a = 0.1;
    float b = 1;
    float c = 4;
    float d = 4;
    float m = -0.5;
    float n = -4;
    float c2 = c * c;
    float c4 = c2 * c2;
    float d2 = d * d;
    float d4 = d2 * d2;
    float d8 = d4 * d4;

    float dist2 = dist * dist;
    float dist4 = dist2 * dist2;
    float dist8 = dist4 * dist4;

    // https://www.desmos.com/calculator/84nyjazqqw
    float potential = m / (a + dist4 / c4) - n / (b + dist4 * dist4 / (d4 * d4));
    float force_mag = 4 * m * dist2 * dist * (d4 - 2 * dist4) / d8;
    return 1.0f * force_mag;
}

// collide two spheres using DEM method
__device__ float3 collideSpheres(float3 posA, float3 posB, float3 velA,
    float3 velB, float radiusA, float radiusB,
    float attraction, float colorScale) {


    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    //float dist2 = dist * dist;
    //float dist4 = dist2 * dist2;
    //float dist5 = dist4 * dist;

    float3 force = make_float3(0.0f);

    /*float a = 0.4;
    float b = 2.8;
    float c = 5;
    float d = 5;
    float g = 0.05;
    float h = 0.44;
    float m = 1;
    float n = 1;*/
    
    /*float potential = a / (dist5 + g) - b / (dist5 + h);
    float force_mag = (b * d * dist4) / ((dist5 + h) * (dist5 + h)) -
                      (a * c * dist4) / ((dist5 + g) * (dist5 + g));*/

    //float force_mag = forceMag_lj(dist);
    float force_mag = forceMag_lj_from_potential(dist * 100);
    //if (dist < attraction * 0.001) {
    //    force_mag = -0.01;
    //}
    //else {
    //    force_mag = 0;
    //}

    //force += 0.1 * (attraction + 0.1) * relPos / (dist + 0.001);
    //float force_mag = 1.2f / (dist4 + 0.05f) - 2.9f / (dist4 + 0.44f);

    //float force_mag = 8 / (dist2 + 0.5) - 5 / ((-1.1 * dist + 0.5) * (-1.1 * dist + 0.5) + 0.5);
    //float force_mag = -1 * 0.4
    force += relPos / dist;
    force *= 0.01 * attraction * force_mag;
    //force -= 0.0000001f * relPos / length(relPos);// *-1 * force_mag;


    force.z = 0.0f;
    return 0.1*force;
}

// collide a particle against all other particles in a given cell
__device__ float3 collideCell(int3 gridPos, uint index, float3 pos, float3 vel,
    float4* oldPos, float4* oldVel, uint* cellStart,
    uint* cellEnd) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)  // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++) {
            if (j != index)  // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius,
                    params.particleRadius, params.attraction, params.colorScale);
            }
        }
    }

    return force;
}

// Fourier transform of partial data
__device__ float3 symmetryCell(int3 gridPos, uint index, float3 pos, float3 vel,
    float4* oldPos, float4* oldVel, uint* cellStart,
    uint* cellEnd) {
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)  // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++) {
            if (j != index)  // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius,
                    params.particleRadius, params.attraction, params.colorScale);
            }
        }
    }

    return force;
}

__device__ float4 velocityToColor(float3 vel) {
    if (length(vel) > 0.000) {
        float mag = vel.x * vel.x + vel.y * vel.y;
        mag *= 1000 * 1000;
        return make_float4(mag, 1.0 - mag, 0.0, 1.0);
    }
    else {
        return make_float4(0.0, 0.5, 0.0, 1.0);
    }
}

__global__ void collideD(
    float4 *newVel,           // output: new velocity
    float4 *newColor,         // output: new color
    float4 *oldPos,           // input: sorted positions
    float4 *oldVel,           // input: sorted velocities
    uint *gridParticleIndex,  // input: sorted particle indices
    uint *cellStart, uint *cellEnd, uint numParticles,
    float colorScale
) {
  uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

  if (index >= numParticles) return;

  // read particle data from sorted arrays
  float3 pos = make_float3(oldPos[index]);
  float3 vel = make_float3(oldVel[index]);

  // get address in grid
  int3 gridPos = calcGridPos(pos);

  // examine neighbouring cells
  float3 force = make_float3(0.0f);
  int neighborDist = 1;
  for (int z = -1; z <= 1; z++) {
    for (int y = -1* neighborDist; y <= neighborDist; y++) {
      for (int x = -1 * neighborDist; x <= neighborDist; x++) {
          if (gridPos.x == 10 || gridPos.y == 5) {
              //continue;
          }
        int3 neighbourPos = gridPos + make_int3(x, y, z);
        force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel,
                             cellStart, cellEnd);
        force -= pos * 0.002; // Center gravity
      }
    }
  }

  // collide with cursor sphere
  //force += collideSpheres(pos, params.colliderPos, vel,
  //                        make_float3(0.0f, 0.0f, 0.0f), params.particleRadius,
  //                        params.colliderRadius, 0.0f);

  // write new velocity back to original unsorted location
  uint originalIndex = gridParticleIndex[index];
  newVel[originalIndex] = make_float4(vel * 1.0 + force, 0.0f);
  newColor[originalIndex] = velocityToColor(vel * colorScale);
}

#endif
