import numpy as np
from . import useful
from . import useful_speedup 

class RandomSpheres:
    def __init__(self, nGrid=100, allow_overlap=True, background=0, label=1, Rs=10, periodic=True):
        self.nGrid = nGrid
        self.background = background
        self.label = label
        self.allow_overlap = allow_overlap
        self.AllowOverlap()
        self.Refresh()
        #self.Rs = None
        #self.nR = None
        self.ListRadii(Rs, nR=None)

    def Refresh(self):
        self.cube = np.zeros((self.nGrid, self.nGrid, self.nGrid))

    def AllowOverlap(self, max_iter=100, periodic=True):
        if self.allow_overlap: self.PutSphere = lambda x,y: OverlappingBall(x,y,periodic=periodic)
        else: self.PutSphere = lambda x,y: NonOverlappingBall(x,y,max_iter=max_iter,periodic=periodic)

    def ListRadii(self, Rs, nR=None):
        if nR is None:
            if type(Rs) in [float, int]: self.Rs = np.array([Rs])
            else: self.Rs = np.array(Rs) if type(Rs) is list else Rs
        else:
            assert type(Rs) in [float, int]
            self.Rs = np.array([Rs for i in range(nR)])
        self.nR = self.Rs.size

    def GetCube_ListRadii(self, refresh=False):
        if self.Rs is None:
            print('Define the radii of the spheres.')
            return None
        if refresh: self.Refresh()
        for i,r in enumerate(self.Rs): 
            self.cube = self.PutSphere(self.cube, r)
            useful.loading_verbose('Number of spheres: %d/%d | Filling fraction:%.3f'%(i+1,self.nR,self.cube.mean()))
        return {'Rs': self.Rs, 'data': self.cube}

    def GetCube_FillingFraction(self, f, refresh=False, multi_rad=False, nOverlapFails=10):
        if self.Rs is None:
            print('Define the radii of the spheres.')
            return None
        if refresh: self.Refresh()
        if self.nR>1 and not multi_rad: 
            print('The radii list contains multiple radii. The run will use the first value only.')
            print('In order to use multiple radii, change {multi_rad} to True.')
        Rs, xi = [], 0
        nOF = 0
        while xi<f:
            r = self.Rs[np.random.randint(0,self.nR)] if multi_rad else self.Rs[0]
            data = self.PutSphere(self.cube, r)
            if data is None:
                print('Cannot find point to put non-overlapping sphere of radius %d.'%r)
                nOF = nOF+1 if multi_rad else nOverlapFails
            else:
                self.cube = data
                Rs.append(r)
                xi = self.cube.mean()
                useful.loading_verbose('Number of spheres: %d | Filling fraction:%.3f'%(len(Rs),xi))
            if nOF>=nOverlapFails: break
        outdata = self.cube
        outdata[outdata==0] = self.background
        outdata[outdata==1] = self.label
        return {'Rs': np.array(Rs), 'data': outdata}


def analytical_powerspect_balls(k, R, xHI):
    Vball = 4*np.pi*R**3/3
    ni    = (1-xHI)/Vball
    kR    = k*R
    WkR   = _W(kR)
    pHI   = (1-xHI)**2*WkR**2/ni
    return pHI

def analytical_bispect_balls(k1, k2, k3, R, xHI):
    Vball = 4*np.pi*R**3/3
    ni    = (1-xHI)/Vball
    WkR1  = _W(k1*R)
    WkR2  = _W(k2*R)
    WkR3  = _W(k3*R)
    bHI   = -(1-xHI)**3*WkR1*WkR2*WkR3/ni**2
    return bHI

def _W(x): return (np.sin(x)-x*np.cos(x))*3/x**3

def OverlappingBall(cube, r, periodic=True):
    centre = np.random.randint(0,cube.shape[0],3)
    return useful_speedup.put_sphere(cube, centre, r, label=1, periodic=periodic)

def NonOverlappingBall(cube, r, max_iter=100, periodic=True):
    count = 0
    while count<max_iter:
        #print(count)
        centre = np.random.randint(0,cube.shape[0],3)
        cube1  = useful_speedup.put_sphere(cube, centre, r, label=1, periodic=periodic)
        cube2  = cube+cube1
        #print(cube2.max())
        if cube2.max()==1: return cube2
        count += 1
    return None


