#include "TH2.h"
#include "TString.h"
// *************************************************
// *** Interpolation of two dimensional histograms *
// *** Author : A.Raspereza (DESY-Hamburg)  ********
// *** based on th1fmorph.cc code by A.Read ********
// *************************************************
                                   
TH2D *th2fmorph(TString chname,
                TString chtitle,
                TH2D * hist1,
                TH2D * hist2,
                float par1,
                float par2,
                float parinterp,
                bool projectOnXaxis = true) ;
  // Input parameters 
  // * chname, chtitle : The ROOT name and title of the interpolated histogram.
  // *                   Defaults for the name and title are "THF1-interpolated"
  // *                   and "Interpolated histogram", respectively.
  // *
  // * hist1, hist2    : The two input histograms.
  // *
  // * par1,par2       : The values of the linear parameter that characterises
  // *                   the histograms (e.g. a particle mass).
  // *
  // * parinterp       : The value of the linear parameter we wish to
  // *                   interpolate to.
  // * projectOnXaxis  : boolean variable, true(false) - interpolation is done
  //                     for TProfile histograms projected on X(Y) axis 


