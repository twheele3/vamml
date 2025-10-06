from copy import deepcopy
from time import sleep

from matplotlib.widgets import Button,Slider,CheckButtons
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# from .. import Experiment
from . import images

def correct_interactive(expt) -> None:
    """Interactively correct alignments using matplotlib widgets. 
    Currently WIP, can't properly invoke JS for it to work in environment.
    This entire code block will work if pasted into a python notebook with 
    proper dependencies loaded, as well as an invocation of %matplotlib widget

     ex:
      ```
      from vamml.batch import Experiment
      from vamml.batchtools.experimental import correct_interactive
      %matplotlib widgets
      expt = Experiment('expt_dir')
      correct_interactive(expt)
      ``` 
      """
    fits = deepcopy(expt.pars['image_fits'])
    i = 0
    shapeidx = fits[i]['best_fit']
    mirror = fits[i]['mirror']
    rotation = fits[i]['rotation']
    imgs = [images.centered_crop(i).resize((expt.pars['array_size'],expt.pars['array_size'])) 
            for i in expt.image_features]
    img = imgs[i]
    shapes = [np.array(images.centered_crop(i).resize((expt.pars['array_size'],
                                                        expt.pars['array_size'])))[...,None] 
            for i in expt.base_shapes]
    b = np.zeros_like(shapes[0])
    shapelen = len(shapes)
    shapes.append(b)
    shapeidx = [shapeidx,shapelen][int(shapeidx is None)]

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    fig.subplots_adjust(bottom=0.25,right=0.85)
    ax.axis('off')
    im = ax.imshow([np.concatenate([
            np.array(img.rotate(-rotation,Image.Resampling.BICUBIC)
                        )[...,None],shapes[shapeidx],b],axis=-1),
            np.concatenate([
                np.array(img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-rotation,Image.Resampling.BICUBIC)
                            )[...,None],shapes[shapeidx],b],axis=-1)][int(mirror)])
    fig.canvas.draw()

    ax_shape  = fig.add_axes((0.215, 0.150, 0.550, 0.030))
    ax_rot    = fig.add_axes((0.900, 0.300, 0.030, 0.550))
    ax_mirror = fig.add_axes((0.830, 0.190, 0.150, 0.050))
    ax_none   = fig.add_axes((0.830, 0.140, 0.150, 0.050))
    axprev    = fig.add_axes((0.400, 0.040, 0.100, 0.075))
    axnext    = fig.add_axes((0.510, 0.040, 0.100, 0.075))
    axreset   = fig.add_axes((0.630, 0.040, 0.120, 0.075))
    axsave    = fig.add_axes((0.770, 0.040, 0.220, 0.075))
    axtxt     = fig.add_axes((0.110, 0.065, 0.100, 0.075))
    axtxt.axis('off')
    t = axtxt.text(0,0,f'Image Index: 0')

    sshape = Slider(
        ax_shape, "Shape", 0, expt.pars['batch_size']-1,
        valinit=shapeidx, valstep=range(expt.pars['batch_size']),
        initcolor='green'  
    )

    srot = Slider(
        ax_rot, "Rotation", 0, 360,
        valinit=rotation, orientation = 'vertical',
        initcolor='none'  
    )

    mirrorcheck = CheckButtons(ax=ax_mirror,labels=['Mirror'],actives=[mirror])
    nonecheck = CheckButtons(ax=ax_none,labels=['No Fit'],actives=[shapeidx is None])

    def updatefig(i):
        shapeidx = fits[i]['best_fit']
        rotation = fits[i]['rotation']
        mirror = fits[i]['mirror']
        img = imgs[i]
        shapeidx = [shapeidx,shapelen][int(shapeidx is None)]
        t.set_text(f'Image Index: {i}')
        im.set_data([np.concatenate([
            np.array(img.rotate(-rotation,Image.Resampling.BICUBIC)
                        )[...,None],shapes[shapeidx],b],axis=-1),
            np.concatenate([
                np.array(img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-rotation,Image.Resampling.BICUBIC)
                            )[...,None],shapes[shapeidx],b],axis=-1)][int(mirror)])
        fig.canvas.draw_idle()

    def updatepars(i):
        srot.eventson = False
        sshape.eventson = False
        nonecheck.eventson = False
        srot.valinit = fits[i]['rotation']
        sshape.valinit = [fits[i]['best_fit'],0][int(fits[i]['best_fit'] is None)]
        srot.reset()
        sshape.reset()
        mirrorcheck.set_active(0,fits[i]['mirror'])
        nonecheck.ignore(nonecheck.set_active(0,fits[i]['best_fit'] is None))
        srot.eventson = True
        sshape.eventson = True
        nonecheck.eventson = True

    class ImgIndex:
        ind = 0
        i = 0

        def update(expt):
            expt.i = expt.ind % len(imgs)
            updatepars(expt.i)
            updatefig(expt.i)

        def next(expt, event):
            expt.ind += 1
            expt.update()

        def prev(expt, event):
            expt.ind -= 1
            expt.update()

    imgcall = ImgIndex()

    def updateshape(val):
        fits[imgcall.i]['best_fit'] = sshape.val
        updatefig(imgcall.i)

    def updaterot(val):
        fits[imgcall.i]['rotation'] = srot.val
        updatefig(imgcall.i)

    def updatemir(val):
        fits[imgcall.i]['mirror'] = mirrorcheck.get_status()[0]
        updatefig(imgcall.i)

    def updatenone(val):
        fits[imgcall.i]['best_fit'] = [0,None][int(nonecheck.get_status()[0])]
        updatepars(imgcall.i)
        updatefig(imgcall.i)
        
    def resetpars(val):
        fits.update(deepcopy(expt.pars['image_fits']))
        updatepars(imgcall.i)
        updatefig(imgcall.i)

    def savepars(val):
        rect=plt.Rectangle((0,0),1,1, transform=fig.transFigure, 
                    clip_on=False, zorder=300, alpha=0.5, color="grey")
        fig.patches.extend([rect])
        expt.pars['image_fits'].update(fits)
        expt.__save_pars()

        fig.canvas.draw()
        sleep(0.2)
        plt.close()
        expt.plot_alignments()

    sshape.on_changed(updateshape)
    srot.on_changed(updaterot)
    mirrorcheck.on_clicked(updatemir)
    nonecheck.on_clicked(updatenone)
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(imgcall.next)
    bprev = Button(axprev, 'Prev')
    bprev.on_clicked(imgcall.prev)
    breset = Button(axreset, 'Reset')
    breset.on_clicked(resetpars)
    bsave = Button(axsave, 'Save&Close')
    bsave.on_clicked(savepars)

    plt.show()