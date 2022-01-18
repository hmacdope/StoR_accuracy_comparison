
#include <math.h>
#include <float.h>
#include <iostream>
#include <iomanip>
#include <random>


typedef float coordinate[3];

#ifdef PARALLEL
  #include <omp.h>
  #define USED_OPENMP 1
#else
  #define USED_OPENMP 0
#endif

void _coord_transform_double_internal(coordinate* coords, int numCoords, double* box)
{
  int i, j, k;
  double newpos[3];
  // Matrix multiplication inCoords * box = outCoords
  // Multiplication done in place using temp array 'new'
  // Used to transform coordinates to/from S/R space in trilinic boxes
#ifdef PARALLEL
#pragma omp parallel for private(i, j, k, newpos) shared(coords)
#endif
  for (i=0; i < numCoords; i++){
    newpos[0] = 0.0;
    newpos[1] = 0.0;
    newpos[2] = 0.0;
    for (j=0; j<3; j++){
      for (k=0; k<3; k++){
        newpos[j] += coords[i][k] * box[3 * k + j];
      }
    }
    coords[i][0] = newpos[0];
    coords[i][1] = newpos[1];
    coords[i][2] = newpos[2];
  }
}


void _coord_transform_float_internal(coordinate* coords, int numCoords, double* box)
{
  int i, j, k;
  float newpos[3];
  // Matrix multiplication inCoords * box = outCoords
  // Multiplication done in place using temp array 'new'
  // Used to transform coordinates to/from S/R space in trilinic boxes
#ifdef PARALLEL
#pragma omp parallel for private(i, j, k, newpos) shared(coords)
#endif
  for (i=0; i < numCoords; i++){
    newpos[0] = 0.0;
    newpos[1] = 0.0;
    newpos[2] = 0.0;
    for (j=0; j<3; j++){
      for (k=0; k<3; k++){
        newpos[j] += coords[i][k] * box[3 * k + j];
      }
    }
    coords[i][0] = newpos[0];
    coords[i][1] = newpos[1];
    coords[i][2] = newpos[2];
  }
}



// creates nrandom floating points between 0 and limit
template <typename T>
void RandomScaledFloatingPoint(T *target, const int nrandom, const int neglimit,
                         const int poslimit, const T scale) {
  std::random_device rd;
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine
  std::uniform_real_distribution<T> distribution(neglimit, poslimit);
  for (size_t i = 0; i < nrandom; i++) {
    target[i] = distribution(gen)/scale;
  }
}

   

int main(){

    std::cout << std::setprecision(10) << std::fixed;
    constexpr bool debugprint = true;
    constexpr int  ncoords = 10000;
    // coordinates are < boxlim in scaled form
    constexpr int boxlim = 30;
    // triclinc boxes in matrix form 30,30,30,45,60,90
    double  dbox[9] = {30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 15.0, 21.213203, 15.0};
    float   fbox[9] = {30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 15.0, 21.213203, 15.0};


   float* scaled_fcoords1 = new float[3*ncoords];
   float* scaled_fcoords2 = new float[3*ncoords];
   float* compare_fabs = new float[3*ncoords];
   double* scaled_dcoords = new double[3*ncoords];
   
   RandomScaledFloatingPoint<float>(scaled_fcoords1, ncoords, 0, boxlim, boxlim);
   // copy 
   for(int i=0; i<3*ncoords; i++) {
       scaled_fcoords2[i] = scaled_fcoords1[i];
   }

  _coord_transform_float_internal((coordinate *)scaled_fcoords1, ncoords, dbox);
  _coord_transform_double_internal((coordinate *)scaled_fcoords2, ncoords, dbox);
   
  float max = 0;
  for(int i=0; i<ncoords; i++) {
      compare_fabs[i] = std::abs(scaled_fcoords1[i] - scaled_fcoords2[i]);
      if constexpr (debugprint) { 
        std::cout << scaled_fcoords1[i] << "  " << scaled_fcoords2[i] << "  " << scaled_fcoords1[i] - scaled_fcoords2[i]  <<  "\n";
      }
      if (compare_fabs[i] > max) {
          max = compare_fabs[i];
      }
      
  }

  std::cout << "MAXIMUM   " <<   max;



}
