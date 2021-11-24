template <typename T>
T MomentMorphing(std::string name,std::string title, const T& h1, const T& h2, double par1, double par2, double par3, double norm = 1.){
//T MomentMorphing(const T& h1, const T& h2){
//
//    std::string name = "test";
//    std::string title = "test";
//    double par1 = 300;
//    double par2 = 400;
//    double par3 = 350;
//    double norm = 1.;

    RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING); // Kill some long RooFit Info log

    /* Store some information about the binning */
    std::vector<double> xs;
    std::vector<double> xbounds;
    std::vector<double> ys;
    std::vector<double> ybounds;
    for(int ibin = 1; ibin <= h1.GetNbinsX(); ++ibin ) {
        xs.push_back(h1.GetXaxis()->GetBinCenter(ibin));
        xbounds.push_back(h1.GetXaxis()->GetBinLowEdge(ibin));
    }
    xbounds.push_back(h1.GetXaxis()->GetBinUpEdge(h1.GetNbinsX()));
    for(int ibin = 1; ibin <= h1.GetNbinsY(); ++ibin ) {
        ys.push_back(h1.GetYaxis()->GetBinCenter(ibin));
        ybounds.push_back(h1.GetYaxis()->GetBinLowEdge(ibin));
    }
    ybounds.push_back(h1.GetYaxis()->GetBinUpEdge(h1.GetNbinsY()));

//    T h3 = T(name.c_str(),title.c_str(),
//                   Nx1,h1.GetXaxis()->GetBinLowEdge(1),h1.GetXaxis()->GetBinUpEdge(Nx1),
//                   Ny1,h1.GetYaxis()->GetBinLowEdge(1),h1.GetYaxis()->GetBinUpEdge(Ny1));

    T h3 = T(name.c_str(),title.c_str(),
                   static_cast<int>(xs.size()),xbounds.data(),
                   static_cast<int>(ys.size()),ybounds.data());

    /* Sanity checks to ensure same binning */
    int Nx1 = h1.GetNbinsX();
    int Ny1 = h1.GetNbinsY();
    int Nx2 = h2.GetNbinsX();
    int Ny2 = h2.GetNbinsY();
    if (Nx1 != Nx2)
        throw std::runtime_error("test");
    if (Ny1 != Ny2)
        throw std::runtime_error("test");
    
    if (h1.GetXaxis()->GetBinLowEdge(1) != h2.GetXaxis()->GetBinLowEdge(1))
        throw std::runtime_error("test");
    if (h1.GetXaxis()->GetBinUpEdge(Nx1) != h2.GetXaxis()->GetBinUpEdge(Nx2))
        throw std::runtime_error("test");
    if (h1.GetYaxis()->GetBinLowEdge(1) != h2.GetYaxis()->GetBinLowEdge(1))
        throw std::runtime_error("test");
    if (h1.GetYaxis()->GetBinUpEdge(Ny1) != h2.GetYaxis()->GetBinUpEdge(Ny2))
        throw std::runtime_error("test");

    if (h1.Integral() <= 0 or h2.Integral() <= 0){
        return h3;
    }

    /* Initialize variables */
    RooRealVar xVar = RooRealVar("xVar","xVar",
                                 *std::min_element(xbounds.begin(), xbounds.end()),
                                 *std::max_element(xbounds.begin(), xbounds.end()));
    RooRealVar yVar = RooRealVar("yVar","yVar",
                                 *std::min_element(ybounds.begin(), ybounds.end()),
                                 *std::max_element(ybounds.begin(), ybounds.end()));

    RooRealVar pVar = RooRealVar("pVar","pVar",par1,par2);

    /* Initialize PDF containers */
    RooArgList listOfMorphs = RooArgList("listOfMorphs");

    TVector paramVec = TVectorD(2);
    paramVec[0] = par1;
    paramVec[1] = par2;

    RooArgList argList = RooArgList(xVar,yVar);
    RooArgSet argSet = RooArgSet(xVar,yVar);

    RooDataHist h1D = RooDataHist((std::string(h1.GetName())+"DataHist").c_str(),
                                  (std::string(h1.GetName())+"DataHist").c_str(),
                                  argList,&h1);
    RooDataHist h2D = RooDataHist((std::string(h2.GetName())+"DataHist").c_str(),
                                  (std::string(h2.GetName())+"DataHist").c_str(),
                                  argList,&h2);

    RooHistPdf h1Pdf = RooHistPdf((std::string(h1.GetName())+"Pdf").c_str(),
                                  (std::string(h1.GetName())+"Pdf").c_str(),
                                  argSet,h1D);
    RooHistPdf h2Pdf = RooHistPdf((std::string(h2.GetName())+"Pdf").c_str(),
                                  (std::string(h2.GetName())+"Pdf").c_str(),
                                  argSet,h2D);

    listOfMorphs.add(h1Pdf);
    listOfMorphs.add(h2Pdf);

    

    /* Build morphing */
    RooMomentMorph morph = RooMomentMorph("morph","morph",
                                pVar,
                                argList,
                                listOfMorphs,
                                paramVec,
                                RooMomentMorph::Linear);
                                //RooMomentMorph::SineLinear);

    /* Create output histogram */
    pVar.setVal(par3);
    double prev = 0.;
    for (int ybin = 0; ybin < ys.size(); ++ybin) {
        for (int xbin = 0; xbin < xs.size(); ++xbin) {
            xVar.setVal(xs[xbin]);
            yVar.setVal(ys[ybin]);
            h3.SetBinContent(xbin+1,ybin+1,morph.getVal(&argSet));
            //std::cout<<"x = ["<<h3.GetXaxis()->GetBinLowEdge(xbin+1)<<","<<h3.GetXaxis()->GetBinUpEdge(xbin+1)<<"]"<<" y = ["<<h3.GetYaxis()->GetBinLowEdge(ybin+1)<<","<<h3.GetYaxis()->GetBinUpEdge(ybin+1)<<"]"<<" -> z = "<<morph.getVal(&argSet)<<" increasing : "<<(h3.GetBinContent(xbin+1,ybin+1) >= prev)<<std::endl; 
            prev = h3.GetBinContent(xbin+1,ybin+1);
        }
    }
    //auto h3 = morph.createHistogram(name.c_str(),xVar,RooFit::Binning(Nx1),RooFit::YVar(yVar,RooFit::Binning(Ny1)));

    h3.Scale(norm/h3.Integral());
    return h3;

}

