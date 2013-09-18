/************************ OMEGA_RHS.INC ******************************/
void rhs(// BH masses
         const FLOAT m1,
	 // BH spins
	 const FLOAT chi1, const FLOAT chi2,
         // state: note all vectors are unit vectors
         const FLOAT omega,
	 const FLOAT S1ux, const FLOAT S1uy, const FLOAT S1uz,
	 const FLOAT S2ux, const FLOAT S2uy, const FLOAT S2uz,
	 const FLOAT LNx,  const FLOAT LNy,  const FLOAT LNz,
         // rhs
         FLOAT *domega,
	 FLOAT *dS1ux, FLOAT *dS1uy, FLOAT *dS1uz,
	 FLOAT *dS2ux, FLOAT *dS2uy, FLOAT *dS2uz,
	 FLOAT *dLNx,  FLOAT *dLNy,  FLOAT *dLNz)
{
  // shorthands
  const FLOAT om=omega;

  // mass constants: total mass M=m1+m2=1.0(!)
  const FLOAT m2=1.0-m1;
  // eta=m1*m2/M^2=m1*m2 for M=1
  const FLOAT eta=m1*m2;
  const FLOAT eta_2=eta*eta;

  // performance abbrevations (minimize pow() function calls)
  const FLOAT om_2=om*om;
  const FLOAT m1_2=m1*m1;
  const FLOAT m2_2=m2*m2;
  const FLOAT om_1_3=pow(om,1./3.);
  const FLOAT om_2_3=om_1_3*om_1_3;
  const FLOAT om_4_3=om_2_3*om_2_3;
  const FLOAT om_5_3=om_4_3*om_1_3;
  const FLOAT om_7_3=om_5_3*om_2_3;
  const FLOAT om_11_3=om_7_3*om_4_3;

  // convert spin magnitude
  const FLOAT u1=chi1*m1_2;
  const FLOAT u2=chi2*m2_2;
  const FLOAT iu1=1./u1;
  const FLOAT iu2=1./u2;

  // actual spin vectors (i.e. not unit)
  const FLOAT S1x=u1*S1ux;
  const FLOAT S1y=u1*S1uy;
  const FLOAT S1z=u1*S1uz;

  const FLOAT S2x=u2*S2ux;
  const FLOAT S2y=u2*S2uy;
  const FLOAT S2z=u2*S2uz;

  // assign rhs quantities
  *domega=19.2*eta*om_11_3*(-om*(.08333333333333333*(chi2*(113.0*
   (m2*m2)+75.0*eta)*(LNz*S2uz+LNy*S2uy+LNx*S2ux)+chi1*(113.0*(m1*m1)+
   75.0*eta)*(LNz*S1uz+LNy*S1uy+LNx*S1ux))-12.56637061435917)-
   .02083333333333333*chi1*chi2*eta*om_4_3*(247.0*(S1uz*S2uz+S1uy*S2uy+
   S1ux*S2ux)-721.0*(LNz*S1uz+LNy*S1uy+LNx*S1ux)*(LNz*S2uz+LNy*S2uy+LNx
   *S2ux))+3.141592653589793*(60.51256613756614*eta_2+
   59.30472883597884*eta-1.094990079365079)*om_7_3-.004674989067841953*
   (15876.0*eta+4159.0)*om_5_3+(3.277777777777778*eta_2+
   6.776289682539683*eta+1.879574514991182)*om_4_3+om_2*
   (-8.152380952380952*log(16.0*om_2_3)+.6037946428571429*eta_2-
   2.162422839506173*(eta*eta*eta)-165.381044105182*eta+
   160.9522690075005)-.002976190476190476*(924.0*eta+743.0)*om_2_3+1.0);

  // rhs for spins
  const FLOAT dS1x=0.5*om_2*S1z*(-3.0*LNy*(LNz*S2z+LNy*S2y+LNx*S2x)+
	S2y+eta*LNy*(3.0*m2/m1+4.0)/om_1_3)-0.5*om_2*S1y*(-3.0*LNz*(LNz*S2z+
	LNy*S2y+LNx*S2x)+S2z+eta*LNz*(3.0*m2/m1+4.0)/om_1_3);

  const FLOAT dS1y=0.5*om_2*S1x*(-3.0*LNz*(LNz*S2z+LNy*S2y+LNx*S2x)+
	S2z+eta*LNz*(3.0*m2/m1+4.0)/om_1_3)-0.5*om_2*S1z*(-3.0*LNx*(LNz*S2z+
	LNy*S2y+LNx*S2x)+S2x+eta*LNx*(3.0*m2/m1+4.0)/om_1_3);

  const FLOAT dS1z=0.5*om_2*S1y*(-3.0*LNx*(LNz*S2z+LNy*S2y+LNx*S2x)+
	S2x+eta*LNx*(3.0*m2/m1+4.0)/om_1_3)-0.5*om_2*S1x*(-3.0*LNy*(LNz*S2z+
	LNy*S2y+LNx*S2x)+S2y+eta*LNy*(3.0*m2/m1+4.0)/om_1_3);

  const FLOAT dS2x=0.5*om_2*(-3.0*LNy*(LNz*S1z+LNy*S1y+LNx*S1x)+S1y+
	eta*LNy*(3.0*m1/m2+4.0)*1.0/om_1_3)*S2z-0.5*om_2*(-3.0*LNz*(LNz*S1z+
	LNy*S1y+LNx*S1x)+S1z+eta*LNz*(3.0*m1*1.0/m2+4.0)/om_1_3)*S2y;

  const FLOAT dS2y=0.5*om_2*(-3.0*LNz*(LNz*S1z+LNy*S1y+LNx*S1x)+S1z+
	eta*LNz*(3.0*m1/m2+4.0)*1.0/om_1_3)*S2x-0.5*om_2*(-3.0*LNx*(LNz*S1z+
	LNy*S1y+LNx*S1x)+S1x+eta*LNx*(3.0*m1*1.0/m2+4.0)/om_1_3)*S2z;

  const FLOAT dS2z=0.5*om_2*(-3.0*LNx*(LNz*S1z+LNy*S1y+LNx*S1x)+S1x+
	eta*LNx*(3.0*m1/m2+4.0)*1.0/om_1_3)*S2y-0.5*om_2*(-3.0*LNy*(LNz*S1z+
	LNy*S1y+LNx*S1x)+S1y+eta*LNy*(3.0*m1*1.0/m2+4.0)/om_1_3)*S2x;

  // rhs for unit spin vectors
  *dS1ux=dS1x*iu1;
  *dS1uy=dS1y*iu1;
  *dS1uz=dS1z*iu1;

  *dS2ux=dS2x*iu2;
  *dS2uy=dS2y*iu2;
  *dS2uz=dS2z*iu2;

  // rhs for unit angular momentum vectors
  *dLNx=-1.0*om_1_3*(dS2x+dS1x)/eta;

  *dLNy=-1.0*om_1_3*(dS2y+dS1y)/eta;

  *dLNz=-1.0*om_1_3*(dS2z+dS1z)/eta;

} // end of rhs()
