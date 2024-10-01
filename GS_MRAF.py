import numpy as np 
from tqdm import trange
import pyfftw


class PhasePattern:
    
    def __init__(self,NX,NY):
        
        self.NX = NX
        self.NY = NY
        assert (NX == NY), 'The grid should be a square.'
        self.x = np.linspace(-NX//2,NX//2,NX)
        self.y = np.linspace(-NY//2,NY//2,NY)
        
        self.X,self.Y = np.meshgrid(self.x,self.y)
        
        self.m = None
        self.A = (np.pi/NX)
        self.B = self.A 
        
        self.phase_slm =  self.A * self.X**2 + self.B * self.Y**2
        self.phase_image = np.zeros((NX,NY)) * 2
        
        self.setROI_masks()
        
    def gaussian_beam(self,w_a):
        beam_amplitude = np.exp( - 2*((self.X)**2 + (self.Y)**2) / (w_a**2))
        
        return beam_amplitude*self.SR
    
    def setInputAmplitude(self,input_amplitude):
        self.input_amplitude = input_amplitude
    
    def setTargetIntensity(self,target_amplitude):
        self.target_amplitude = np.sqrt(abs(target_amplitude))

    def setROI_masks(self):
        
        ROI = np.zeros((self.NX,self.NY))
        ROI = self.define_square_mask(ROI, self.NX//2, i_pos = self.NX//2 , j_pos = self.NX//2)
        
        self.SR = ROI 
        self.NR = np.ones((self.NX,self.NY)) - self.SR
        
        return None
        
    def setMRAF_masks(self,size,shape = 'square'):
        MRAF_ROI = np.zeros((self.NX,self.NY))
        
        if shape == 'circular':
            MRAF_ROI = self.define_round_mask(MRAF_ROI, radius = size, i_pos = self.NX//2, j_pos = self.NX//2)
        else:
            MRAF_ROI = self.define_square_mask(MRAF_ROI, length = size , i_pos = self.NX//2, j_pos = self.NX//2)
            
        self.MRAF_SR = MRAF_ROI
        self.MRAF_NR = np.ones((self.NX,self.NY)) - self.MRAF_SR
    
    
    
    def define_square_mask(self,img,length,i_pos,j_pos):
    
        mid=(int)(length/2)
        i_int=i_pos-mid+np.arange(length)
        j_int=j_pos-mid+np.arange(length)
    
        img[np.ix_(i_int, j_int)]= np.ones((length,length))
        
        return img
    
    def define_round_mask(self,img,radius,i_pos,j_pos):
        
        area = (self.X - i_pos)**2 + (self.Y - j_pos)**2 < radius**2
        
        img[area] = 1 
        
        return img 
    
    
    def add_point(self,img,length,i_pos,j_pos):
        '''
        
        assert(len(diam)==len(i_pos))
        assert(len(i_pos)==len(j_pos))
        '''
        mid=(int)(length/2)
        i_int=i_pos-mid+np.arange(length)
        j_int=j_pos-mid+np.arange(length)

        img[np.ix_(i_int, j_int)]= np.ones((length,length))
        
        return None 
        
    def normalize(self,img):
        img = np.abs(img)
        
        return (img-np.min(img))/(np.max(img)+np.min(img))
    

    
    def getUniformity(self,img):
        
        
        
        mini=np.min(np.abs(img))
        maxi=np.max(np.abs(img))
        
        uniformity =1-(maxi-mini)/(maxi+mini)
        
        
        return uniformity
    
    def getRMSE(self,predictions,targets):
        
        rmse = np.sqrt(((predictions - targets) ** 2).mean())
        
        return rmse
        

    def SLM_TO_IMAGE(self,field):
        
        out_field = pyfftw.interfaces.numpy_fft.fft2(field,norm = 'ortho')
        out_field = np.fft.fftshift(out_field)
        
        return out_field
    
    def IMAGE_TO_SLM(self,field):
        
        out_field = np.fft.ifftshift(field)
        out_field = pyfftw.interfaces.numpy_fft.ifft2(out_field,norm = 'ortho')
        
        return out_field
    
    def constraint_image_plane(self,field,w):
        
        phase = np.angle(field)
        amplitude = np.abs(field)
        
        
        U = self.m * self.normalize(self.MRAF_SR * self.target_amplitude) + (1-self.m) * self.normalize(self.MRAF_NR * amplitude)
        
        out_field = self.SR * U * np.exp(1j *phase *self.SR)
        
        return out_field

    
    def ComposeSignal(self,amplitude,phase):
        signal = amplitude*np.exp(1j*phase)
        return signal
    
    def DecomposeSignal(self,field):
        
        amplitude=np.abs(field)
        phase=np.angle(field)
        
        return (amplitude,phase)
    
    def update_weights(self,img,w_prev):
        
        w =  np.ones((self.NX,self.NY))
        
        target = self.normalize(self.target_amplitude[self.target_amplitude >0])
        img = self.normalize(img[self.target_amplitude >0 ]) + 1e-9
        
        
        w[self.target_amplitude >0] = np.sqrt( target/img) * w_prev[self.target_amplitude >0] 
        
        
        return w 

    def PhaseRetrieve(self,n,m):
        self.m  = m 
        rmse = np.zeros(n)
        uniformity = np.zeros(n)
        
        current_uniformity = 0
        current_error = 0
        
        
        w = np.ones((self.NX,self.NY))
            
        for i in trange(n):
            
            slm_field = self.ComposeSignal(self.input_amplitude*self.SR,self.phase_slm*self.SR)
            image_field = self.SLM_TO_IMAGE(slm_field)
            
            
            reconstructed_image,phase_image = self.DecomposeSignal(image_field)
            #w = self.update_weights(reconstructed_image,w_prev = w) #TBD
            image_field = self.constraint_image_plane(image_field,w)
            
        
            slm_field = self.IMAGE_TO_SLM(image_field)
            _,self.phase_slm = self.DecomposeSignal(slm_field)
            
            current_error = self.getRMSE(reconstructed_image*self.MRAF_SR,self.target_amplitude*self.MRAF_SR)
            current_uniformity = self.getUniformity((reconstructed_image[self.target_amplitude == 1]))
            
            rmse[i] = current_error
            uniformity[i] = current_uniformity 
                    
                
                
        reconstructed_image = self.normalize(reconstructed_image)   
        
        self.phase_slm = self.convert_to_256_map(self.phase_slm[self.SR.astype(bool)].reshape((self.NX//2,self.NY//2)))
        self.metric = {'rmse':rmse,'uniformity':uniformity}
        self.reconstructed_image = reconstructed_image
        
        
        return (self.reconstructed_image,self.phase_slm,w)
    

    def convert_to_256_map(self,slm_pattern):
        """
        Convert phase pattern to a 256-level map.
    
        Parameters:
            slm_pattern (2D numpy array): SLM pattern.
    
        Returns:
            slm_map (2D numpy array): 256-level phase map.
        """
        slm_map = np.round((slm_pattern - slm_pattern.min()) / (slm_pattern.max() - slm_pattern.min()) * 255)
        
        return slm_map.astype(np.uint8)
    
    def pad(self,img):
        
        L = img.shape[0]
        tmp = np.zeros((2 * L, 2 * L))
        tmp[int(2 * L / 4):int(3 * 2 * L / 4), int(2 * L / 4):int(3 * 2 * L / 4)] = img
        img = tmp
        
        return img 


