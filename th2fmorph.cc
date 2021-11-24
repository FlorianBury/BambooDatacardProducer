#include "th2fmorph.h"
//#include "th1fmorph.h"
#include "TROOT.h"
#include "TH1F.h"
#include <iostream>
#include <cmath>

// *************************************************
// *** Interpolation of two dimensional histograms *
// *** Author : A.Raspereza (DESY-Hamburg)  ********
// *** based on th1fmorph.cc code by A.Read ********
// *************************************************

TH2D *th2fmorph(TString chname, 
        TString chtitle,
        TH2D *hist1,TH2D *hist2,
        float par1, float par2, float parinterp,
        bool projectOnXaxis) {
 
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
 
  int nBinsX1 = hist1->GetNbinsX();
  int nBinsX2 = hist2->GetNbinsX();
  int nBinsY1 = hist1->GetNbinsY();
  int nBinsY2 = hist2->GetNbinsY();

  if (nBinsX1!=nBinsX2) {
    std::cout << "Two histograms have different number of bins along X axis! Returning NULL pointer..." << std::endl;
    return NULL;
  }
  if (nBinsY1!=nBinsY2) {
    std::cout << "Two histograms have different number of bins along Y axis! Returning NULL pointer..." << std::endl;
    return NULL;
  }

  float xMin1 = hist1->GetXaxis()->GetBinLowEdge(1);
  float xMin2 = hist2->GetXaxis()->GetBinLowEdge(1);
  float xMax1 = hist1->GetXaxis()->GetBinLowEdge(1+nBinsX1);
  float xMax2 = hist2->GetXaxis()->GetBinLowEdge(1+nBinsX1);

  if (float(TMath::Abs(xMin1-xMin2))>1e-5) {
    std::cout << "Two histograms have different lower bounds of X axis! Returning NULL pointer..." << std::endl;
    return NULL;
  }

  if (float(TMath::Abs(xMax1-xMax2))>1e-5) {
    std::cout << "Two histograms have different upper bounds of X axis! Returning NULL pointer..." << std::endl;
    return NULL;
  }

  float yMin1 = hist1->GetYaxis()->GetBinLowEdge(1);
  float yMin2 = hist2->GetYaxis()->GetBinLowEdge(1);
  float yMax1 = hist1->GetYaxis()->GetBinLowEdge(1+nBinsY1);
  float yMax2 = hist2->GetYaxis()->GetBinLowEdge(1+nBinsY1);
    
  if (float(TMath::Abs(yMin1-yMin2))>1e-5) {
    std::cout << "Two histograms have different lower bounds of Y axis! Returning NULL pointer..." << std::endl;
    return NULL;
  }

  if (float(TMath::Abs(yMax1-yMax2))>1e-5) {
    std::cout << "Two histograms have different upper bounds of Y axis! Returning NULL pointer..." << std::endl;
    return NULL;
  }

  if (float(TMath::Abs(par2-par1))<1e-5) {
    std::cout << "Boundary parameters do not differ! Return first histogram (hist1)..." << std::endl;
    return hist1;
  }

  float koeff = (parinterp-par1)/(par2-par1);

  float xbins[1000];
  float ybins[1000];
  float bins[1000];

  for (int iB=0; iB<=nBinsX1; ++iB)
    xbins[iB] = hist1->GetXaxis()->GetBinLowEdge(iB+1);

  for (int jB=0; jB<=nBinsY1; ++jB)
    ybins[jB] = hist1->GetYaxis()->GetBinLowEdge(jB+1);

  TH2D * interpolated2d = new TH2D(chname,chtitle,nBinsX1,xbins,nBinsY1,ybins);

  int nRefBins = nBinsX1;
  int nProjBins = nBinsY1;

  if (projectOnXaxis) {
    nRefBins = nBinsY1;
    nProjBins = nBinsX1;
    for (int iB=0; iB<=nProjBins; ++iB) { 
      bins[iB] = xbins[iB];
    }
  }
  else {
    for (int iB=0; iB<=nProjBins; ++iB) {
      bins[iB] = ybins[iB];
    }
  }

  for (int iB=0; iB<nRefBins; ++iB) {
    TH1F * hist1D1 = new TH1F("hist1D1","",nProjBins,bins);
    TH1F * hist1D2 = new TH1F("hist1D2","",nProjBins,bins);
    //    std::cout << "OK3 " << std::endl;
    //    int rh;
    //    std::cin >> rh;
    float norm1 = 0;
    float norm2 = 0;
    for (int jB=0; jB<nProjBins; jB++) {
      int first = iB + 1;
      int second = jB + 1;
      if (projectOnXaxis) {
    first = jB + 1;
    second = iB + 1;
      }
      float  x1 = hist1->GetBinContent(first,second);
      float ex1 = hist1->GetBinError(first,second);
      hist1D1->SetBinContent(jB+1,x1);
      hist1D1->SetBinError(jB+1,ex1);
      float  x2 = hist2->GetBinContent(first,second);
      float ex2 = hist2->GetBinError(first,second);
      hist1D2->SetBinContent(jB+1,x2);
      hist1D2->SetBinError(jB+1,ex2);
      norm1 += x1;
      norm2 += x2;
    }
    //    std::cout << "OK4 " << std::endl;
    //    std::cin >> rh;
    
    float normProj = norm1 + (norm2-norm1)*koeff;
    TH1F * histInterpolated = NULL; 
    if (norm1<1e-10 || norm2<1e-10) { // use linear interpolation if one of the histos is empty 
      histInterpolated = new TH1F("HistoProj","histoProj",nProjBins,bins);
      for (int k=0; k<nProjBins; ++k) {
    float x1bin = hist1D1->GetBinContent(k+1);
    float x2bin = hist1D2->GetBinContent(k+1);
    float xc = x1bin + (x2bin-x1bin)*koeff;
    if (xc<0) {
      std::cout << "Warning: bin content of interpolated histogram < 0 : " << xc << std::endl;
      xc = 0;
    }
    histInterpolated->SetBinContent(k+1,xc);
    //std::cout<< "norm -> 0   ...   one histo seems to be empty" <<std::endl;
      }
    }
    else {
      TH1F * clone1 = new TH1F("clone1","",nProjBins,bins);
      TH1F * clone2 = new TH1F("clone2","",nProjBins,bins);
      clone1->Add(hist1D1,hist1D1,1/norm1,0.);
      clone2->Add(hist1D2,hist1D2,1/norm2,0.);
      histInterpolated = th1fmorph("HistoProj","histoProj",
                   clone1,clone2,
                   par1,par2,parinterp,
                   1,0); //felix von 0 auf 3
      delete clone1;
      delete clone2;
      histInterpolated->Scale(normProj);
      //      int nBB = histInterpolated->GetNbinsX();
      //      std::cout << "reference bin : " << iB+1 <<  "  nBins = " <<  nBB
      //        << "    xlow = " << histInterpolated->GetBinLowEdge(1) 
      //            << "    xup  = " << histInterpolated->GetBinLowEdge(nBB+1) << std::endl;
    }
    for (int jB=0; jB<nProjBins; ++jB) {
      float x  = histInterpolated->GetBinContent(jB+1);
      float ex = histInterpolated->GetBinError(jB+1);
      int first = iB + 1;
      int second = jB + 1;
      if (projectOnXaxis) {
    first = jB + 1;
    second = iB + 1;
      }
      interpolated2d->SetBinContent(first,second,x);
      interpolated2d->SetBinError(first,second,ex);
    }
    //    std::cout << "OK1" << std::endl;
    //    int h;
    //    std::cin >> h;
    delete histInterpolated;
    delete hist1D1;
    delete hist1D2;
    //    std::cout << "OK2" << std::endl;
    //    int h;
    //    std::cin >> h;
    
  }

  return interpolated2d;

}




