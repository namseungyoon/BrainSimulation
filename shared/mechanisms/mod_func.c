#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _DetAMPANMDA_reg();
extern void _DetGABAAB_reg();
extern void _ProbAMPANMDA_EMS_reg();
extern void _ProbGABAAB_EMS_reg();
extern void _VecStim_reg();
extern void _cacum_reg();
extern void _cagk_reg();
extern void _cal_reg();
extern void _can_reg();
extern void _cat_reg();
extern void _hd_reg();
extern void _kad_reg();
extern void _kap_reg();
extern void _kca_reg();
extern void _kdb_reg();
extern void _kdr_reg();
extern void _kdrb_reg();
extern void _kmb_reg();
extern void _na3_reg();
extern void _nax_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," DetAMPANMDA.mod");
fprintf(stderr," DetGABAAB.mod");
fprintf(stderr," ProbAMPANMDA_EMS.mod");
fprintf(stderr," ProbGABAAB_EMS.mod");
fprintf(stderr," VecStim.mod");
fprintf(stderr," cacum.mod");
fprintf(stderr," cagk.mod");
fprintf(stderr," cal.mod");
fprintf(stderr," can.mod");
fprintf(stderr," cat.mod");
fprintf(stderr," hd.mod");
fprintf(stderr," kad.mod");
fprintf(stderr," kap.mod");
fprintf(stderr," kca.mod");
fprintf(stderr," kdb.mod");
fprintf(stderr," kdr.mod");
fprintf(stderr," kdrb.mod");
fprintf(stderr," kmb.mod");
fprintf(stderr," na3.mod");
fprintf(stderr," nax.mod");
fprintf(stderr, "\n");
    }
_DetAMPANMDA_reg();
_DetGABAAB_reg();
_ProbAMPANMDA_EMS_reg();
_ProbGABAAB_EMS_reg();
_VecStim_reg();
_cacum_reg();
_cagk_reg();
_cal_reg();
_can_reg();
_cat_reg();
_hd_reg();
_kad_reg();
_kap_reg();
_kca_reg();
_kdb_reg();
_kdr_reg();
_kdrb_reg();
_kmb_reg();
_na3_reg();
_nax_reg();
}
