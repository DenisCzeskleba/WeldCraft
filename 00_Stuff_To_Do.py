"""
Do you want to adjust the following here? maybe do it in the actual WeldCraft version?

Adjust last weld bead options

redo the saving logic:
    - add some way to accumulate date and write it all at once. i.e. 500mb of stuff, then write, maybe different thread
    - safe ALL the meta data
    - while saving that meta data, maybe think about the format of saving, i.e. seperate data sets,
      append vs new or something like that
    - move to seperate function? could be cleaner main script
    - have two different functions for meta data and main data
    - have another look at the slow diffusion saving logic. i dont think we have a way to save infrequently?

Implement the stuff with chemical drive (not just concentration difference)
    - Optional setting? It might be much slower (50%?) so maybe we can activate if necessary

add the option to start new bead for some temperature

add the convection ... ?? from the heat version ??

add weld styles - Kjell, ISO3690, angle for base metal flanks?,

while youre at it, allow for batch start! so adjust the following to be externally definable
    - names for files
    - thickness (and everything that goes with that)
    - Style of weld? So the one we have, Kjells thing and maybe iso3690
    - t85? kind of together with thickness

BIG ONE - fix the boundary stuff
Need to do this to allow for different hydro diffusion coefficients too!

------------------------------------------------------------------------------------------------------------------------

working on switching to actual time for main loop:
you changed the change times generation function to return actual times
you also partly added the option to add weld bead via interpass temperature
you need to take care of the make_animation thing too, it still wants to use steps


------------------------------------------------------------------------------------------------------------------------
Stuff I wrote before? maybe parts are still useful? I dont remember when or why i vomited this stuff into some code

REMEMBER:   The assumption within areas of the same diffusion coefficient is a uniform spreading of the TEMPERATURE
                or hydrogen CONCENTRATION. This does lend itself well to the established models, especially for hydrogen
                diffusion because the hydrogen diffusivity coefficient is most commonly measured and used. But it only
                models concentration e.i. the Dcoeff = X ![m²/s]!, which is the equivalent of temperature but NOT HEAT!

                Thus, this can not adequately describe the boundary interactions between interacting areas of differing
                diffusivity coefficients. Consider for example the following:
                A steel to air boundary and a simple heat transfer, ignoring convection i.e. completely still air in a
                perfectly sealed system and no radiation.
                The Thermal Diffusivity [m²/s] for air and steel are virtually the same at 15-20. Using the thermal
                diffusivity would give us a cooling for the metal plate from 200°C to 175°C and heating for air from
                25°C to 50°C in a given time for example. But even common sense tells us that while the air above the
                metal plate, might heat up from 25°C to 50°C, the steel plate may only cool from 200°C to 199°C. Using
                diffusivity would essentially result in energy getting destroyed as per volume (which a simulation mesh
                represents) the steel plate would lose much more energy than the air would gain. This means that a
                caloric calculation of diffusion over the boundary is necessary to account for the law of conservation
                of energy for temperature diffusion, or conservation mass in the case of hydrogen diffusion.
                This stems from different densities (ρ), specific heat capacities (c) and thermal conductivities (λ)
                for steel and air.
                Air: ρ = 0,0013 kg/dm², c = 1.01 [kJ/kgK], λ = 0.026 [W/mK]
                Steel: ρ = 7.85 kg/dm², c = 0.465 [kJ/kgK], λ = 75 [W/mK]
                With D = λ / (ρ*c) --> D_Air = ~20, D_Steel = ~20

                This also leads to several implications for the simulation. Especially for the finite difference model.
                Because for example, what happens when 3 areas of different Materials meet? We can not avoid these
                geometries via a clever application of a mesh in FDM. But the reverse is also true, if the materials are
                very similar, we might ignore minor differences in material characteristics in macro simulations, like
                base metal to heat affected zone to weld metal. One should, in that case take care to UNDERestimate the
                heat/temperature in the system though, because ultimately we are interested in the hydrogen distribution
                of the weld. And a lower overall temperature would lead to an overestimation of the hydrogen
                concentration, which would essentially lead to a built in safety factor of the simulation. This might
                even become necessary as exact values for specific microstructures/alloy compositions are unknown.

                One might also ask why the simulation would not simply calculate the energy or mass, and calculate back
                to temperature or concentration after the fact, for analysis and visualization. This is a deliberate
                choice to facilitate the intuitive use of this program / simulation. As the main aim of this simulation
                is to PREDICT location and concentration of temperatures and hydrogen concentration when variables
                like heat input, diffusion coefficients, bead size, wait times etc. get varied. This can only be done
                reliably if what the program does, makes intuitive sense for the large majority of the code. Otherwise
                it would end up as a neat toy to play around with until pretty pictures emerge. Unless one is already
                knee-deep into numerics, meshes etc.. And if you are reading this, and are that person, feel free to
                adjust the solver (Crank-Nicolson should be partially implemented already and just commented out,
                remember, bigger dt's are allowed now.)

                The steel to air example is simple and intuitive enough that eventhough the numeric calculations on the
                boundary are complicated, one can readily convince oneself that what is calculated in 2 or 3D in a
                complex tensor, is just like the example, just a little more complicated.
                This also explains why FDM was chosen over implicit solvers, FEM or FVM etc.: To allow an intuitive
                understanding. This by no means should be interpreted as FDM being less accurate (though it can be)
                or as this simulation being less than. It is a single purpose tool, aimed at precision for a very narrow
                application window. It does what it does and it does it really well for what it is designed for. By its
                very nature, this highly specific application also offers significant optimization advantages over
                general/ generic programs such as ANSYS, while offering the needed customization for this specific task.
                No optimization aside from NumPy vectorization, such as explicit parallelization, GPU usage etc. was
                implemented to alllow for use on any system that can run a python environment. Readability, and
                intuitive understanding and use, took priority over optimization. If you know what you are doing and
                have access to CUDA: try sending the tensors to your GPU and only calling back when you want to save,
                this should result in about a 50x speed boost, depending on matrix size (bigger is better).
"""
