from typing import Optional
from pymatgen.core import Lattice, Structure
import numpy as np

#TODO: Refactor names of functions
def LatticeCordinatFromPFunitCell(length: float,
                                  indLattice: int,
                                  precent_dopping: float,
                                  width:Optional[float] = None,
                                  height:Optional[float] = None,
                                  host_material:int = 'NaYF4',
                                  shape:Optional[str] = "Sphere"):
    """
    The main function that generate a lattice according the primitive unit cell chosen material (now 1: WSe2 or 2: NaYF4 3: NaYF4_disorder)
    The crystallattice generated is 2m+2 by 2n+2 by 2l+2 of the primitive unit cells
    shape=="sphere cut to a sphere with radious=Length otherwise makes a 3D rectangular acording to Length, Width, heigh
    if you wish to add dopping use precent_dopping to assign the presentage amount of the chosen lattice site, which is chosen according to indLattice

    :param length: desired length of nanoparticle in nm, unless shapeChoice = "Sphere", in which case the length is the radius of the sphere
    :param width: width of simulation cell. Ignored if shapeChoice = "Sphere,"
    :param height: height of simulation cell. Ignored if shapeChoice = "Sphere,"
    :param host_material: unit cell. If 1, then WSe2. If 2, then NaYF4.  Otherwise, make it disordered NaYF4
    :param shape: "sphere" to cut a sphere with radius=Length; otherwise makes a 3D rectangular acording to Length, Width, height
    :param precent_dopping: percent doping of lattice site with index indlattice
    :param indLattice:index of lattice site to dope (see example above)
    :return:

    examples:
        LatticeCordinatFromPFunitCell(2,2,2,1,"Sphere",0,1)-->WSe2 R=2 sphere, no dopping
        LatticeCordinatFromPFunitCell(2,2,2,2,"Sphere",0.3,1)-->NaYF4 R=2 sphere, doppingb30% out of Y sites (second index)
        LatticeCordinatFromPFunitCell(2,1,1,2,"rectangular",0.1,2)--> NaYF4 rectangular l,m,n 2,1,1, doppingb10% out of F sites (third index)
    """
    # Generate starting unit cell
    if host_material == 'WSe2':
        lattice = Lattice.hexagonal(a=3.327, c=15.069)
        species = ['Se', 'Se', 'Se', 'Se', 'W', 'W']
        positions = [[0.3333, 0.6667, 0.6384], [0.3333, 0.6667, 0.8616], [0.6667, 0.3333, 0.1384],
                     [0.6667, 0.3333, 0.3616], [0.3333, 0.6667, 0.25], [0.6667, 0.3333, 0.75]]
        struct = Structure(lattice, species=species, coords=positions)
    elif host_material == 'NaYF4':
        lattice = Lattice.hexagonal(a=6.067, c=7.103)
        species = ['Na', 'Na', 'Na', 'Y', 'Y', 'Y', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F']
        positions = [[0.3333, 0.6667, 0.5381], [0.3333, 0.6667, 0.9619], [0.6667, 0.3333, 0.75], [0, 0, 0.9969],
                     [0, 0, 0.5031], [0.6667, 0.3333, 0.25], [0.0272, 0.2727, 0.2500], [0.0572, 0.2827, 0.7500],
                     [0.2254, 0.9428, 0.7500], [0.2455, 0.9728, 0.2500], [0.4065, 0.3422, 0.0144],
                     [0.4065, 0.3422, 0.4856], [0.6578, 0.0643, 0.0144], [0.6578, 0.0643, 0.4856],
                     [0.7173, 0.7746, 0.7500], [0.7273, 0.7545, 0.2500], [0.9357, 0.5935, 0.0144],
                     [0.9357, 0.5935, 0.4856]]
        struct = Structure(lattice, species=species, coords=positions)
    else:
        #TODO: Generate disordered NaYF4
        raise NotImplementedError("disordered NaYF4 not implemented")

    # Transform unit cell into desired nanoparticle (size and shape)
    if shape == "sphere":
        _struct = struct.copy()

        _struct.make_supercell(np.ceil(np.divide(length * 2.45, struct.lattice.abc)))

        cart_coords = _struct.cart_coords
        center = np.divide(np.sum(_struct.lattice.matrix, axis=0), 2)

        translated_coords = np.subtract(cart_coords, center)
        abs_coords = np.abs(translated_coords)
        distance = np.sqrt(np.sum(np.square(abs_coords), axis=1))
        i_sites_in_sphere = np.where(distance <= length)[0]
        len(i_sites_in_sphere)
        sites = []
        for i in i_sites_in_sphere:
            sites.append(_struct.sites[i])
        nano_particle = Structure.from_sites(sites)
    else:
        #TODO: Implement other shapes such as rectangle or hexagonal rod/plate
        raise NotImplementedError(f"{shape} not implemented")

    # Add dopants to the nanoparticle

    return nano_particle

# Disordered NaYF4
	# elseif(stringmatch(W_ofNamesU[0], "Na_4e_abc")&& stringmatch(W_ofNamesU[1], "Y_2d_abc")&& stringmatch(W_ofNamesU[2], "F_6h_abc"))
	# 	make/O/T/N=3 W_ofName2P="";
    #
	# 	wave WV2=$W_ofNamesU[1]
	# 	WV2={{0.3333, 0.6667, 0.7500},{0.6667, 0.3333, 0.2500}};//Y_2d_abc, lattice coordinates of yttrium atoms only
	# 	MatrixOp/O WV2=WV2^t
	# 	coversionMat(90,90,120, 5.9688/10, 5.9688/10, 3.5090/10);//deg,nm
	# 	wave conversion_mat
	# 	wave m_n_l= SizeCalcPC(Length, Width, height,conversion_mat)
    #
	# 	GenerateLattice(WV2,m_n_l[0],m_n_l[1],m_n_l[2],conversion_mat) //generates M_xyz
	# 	wave M_xyz //sorted 2D wave containing rows of atomic coordinates for lattice
	# 	duplicate/O M_xyz, Y_d_xyz
	# 	wave M_xyz_2=Y_d_xyz
	# 	ArrangeDisorderRatio(M_xyz_2, "Y",0.75)//arrange disorder	1:3, generates XYZ_Ytemp wave AND NaToadd_temp
	# 	wave XYZ_Ytemp, NaToadd_temp
	# 	duplicate/O XYZ_Ytemp, M_xyz_2; //take new Y wave with disorder
	# 	if (stringmatch(shapechoice, "Sphere"))// sphere
	# 		CutLattice2Sphere(M_xyz_2,Length,PlotBoll)// xyz mat, radious and plottingBool
	# 		wave XYZ_Sphere2=$(nameofwave(M_xyz_2)+ "_Sphere")
	# 		W_ofName2P[1]=nameofwave(XYZ_Sphere2)
	# 	else
	# 		W_ofName2P[1]=nameofwave(M_xyz_2)
	# 	endif
    #
    #
	# 	wave WV1=$W_ofNamesU[0]
	# 	WV1={{0,0,0.0950},{0,0,0.4050},{0,0,0.5950},{0,0,0.9050}};//Na_4e_abc, relative lattice coordinates of sodium ions only
	# 	MatrixOp/O WV1=WV1^t
    #
	# 	GenerateLattice(WV1,m_n_l[0],m_n_l[1],m_n_l[2],conversion_mat) //generates M_xyz
	# 	duplicate/O M_xyz, Na_d_xyz
	# 	ArrangeDisorderRatio(Na_d_xyz, "Na",0.75)//arrange disorder	1:3, generates XYZ_Natemp wave
	# 	wave XYZ_Natemp;
	# 	Concatenate/O/NP=0 {XYZ_Natemp,NaToadd_temp},Na_d_xyz //adds Na coords from this site to the Na fractionally occupied sites created when dealing with Y above (NaToAdd_temp)
	# 	wave M_xyz_1=Na_d_xyz
	# 	print dimsize(M_xyz_1,0),dimsize(XYZ_Natemp,0),dimsize(NaToadd_temp,0)
	# 	if (stringmatch(shapechoice, "Sphere"))// sphere
	# 		CutLattice2Sphere(M_xyz_1,Length,PlotBoll)// xyz mat, radious and plottingBool
	# 		wave XYZ_Sphere1=$(nameofwave(M_xyz_1)+ "_Sphere")
	# 		W_ofName2P[0]=nameofwave(XYZ_Sphere1)
	# 	else
	# 		W_ofName2P[0]=nameofwave(M_xyz_1)
	# 	endif
    #
    #
    #
	# 	wave WV3=$W_ofNamesU[2]
	# 	WV3={{0.0849,0.6834,0.2500},{0.3166,0.4015,0.2500},{0.4015,0.0849,0.7500},{0.5985,0.9151,0.2500} ,{0.6834,0.5985,0.7500} ,{0.9151,0.3166,0.7500}};//F_d_abc
	# 	MatrixOp/O WV3=WV3^t
	# 	GenerateLattice(WV3,m_n_l[0],m_n_l[1],m_n_l[2],conversion_mat)
	# 	duplicate/O M_xyz, F_d_xyz
	# 	wave M_xyz_3=F_d_xyz
	# 	if (stringmatch(shapechoice, "Sphere"))// sphere
	# 		CutLattice2Sphere(M_xyz_3,Length,PlotBoll)// xyz mat, radious and plottingBool
	# 		wave XYZ_Sphere3=$(nameofwave(M_xyz_3)+ "_Sphere")
	# 		W_ofName2P[2]=nameofwave(XYZ_Sphere3)
	# 	else
	# 		W_ofName2P[2]=nameofwave(M_xyz_3)
	# 	endif

def random_lattice_doping():
    """
    //************************************************************************************************
    // Function RandomLatticeDopping(XYZ_loc, precent_dopping)
    // Generate randomly distributed dopants within a given lattice corrdinates (XYZ_loc) according to the requested dopping precentage
    //
    // OUTPUTS:
    //		XYZ_corrDopants: Makes 2D wave XYZ_corrDopants that gives the coordinates of the doped positions only
    //		XYZ_loc: DELETES doped positions from input XYZ_loc wave
    //
    //************************************************************************************************
    :return:
    """
    #TODO: Implement a random doping
    pass


def arrange_disorder_ratio():
    """
    //***********************************************************************
    //ArrangeDisorderRatio(Wave_XYZtemp, stringName,probability_dis)
    // arrange disorder for NaYF4_disorder
    //randomize position according to a given probability
    //***********************************************************************

    :return:
    """
    #TODO: Implement this function
    pass