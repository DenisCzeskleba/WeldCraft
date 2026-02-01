"""
This document is a work in progress for now. Basically it describes how this program could be used
to determine the hydrogen distribution, needed DHT or needed welding parameters.

For any steel weld (S690 in this example) you obviously operate in some process window.
Say t8/5 between 5 and 20 s. For our welding process we used SAW and 4mm wire, so our practically achievable lower
limit was 10s t8/5 and we limited our experiments to 20s t8/5 for the upper limit as per the filler materials producers
recomendation. So thats your "region of interest" if you wanted to explore this aspect. For us, we were interested
in the hydrogen diffusion i.e. to avoid hydrogen assisted cold cracking, by exploring a possible DHT.

First, you'll have to input the dimensions of your weld specimen. Keep in mind to simplify as much as possible. Is the
weld symmetric? Do you really need to simulate both sides of the weld? Is the real/full length of the weld seam
important? How about the width? Do you really need to simulate the 15°C change 100mm away from the weld seem? This not
only impacts calculation time but also makes calibration much easier. It is also good practice to simulate only whats
necessary, unless you have good reason not to. For example if you want fancy pictures/video/animation of your simulation
for communication purposes. In these cases, obviously visualization takes precident over common conventions or
calculation time optimizations. For reference: A full weld process simulation (welding, cooling, 5 days wait time),
including the temperature dependent hydrogen diffusion of a 150x100mm area (0.5mm mesh) with temperature dependent time
discretization (faster diffusion, shorter time steps needed) takes about 15 minutes to simulate. So don't stress too
much about this sort of optimization. Also, make sure you don't over discretize the mesh. NO ONE needs 0.1mm accuracy
and even 0.5mm is likely overkill. Remember that using a 5 times smaller mesh means you have 25 times as many cells.
But not only that, you also have to simulate many more time steps AND every iteration takes longer due to memory
read/write and/or saving to disk limitations. Rule of thumb is roughy: cube the increase and multiply by 2.

The second thing one will need to do is to figure out how to simulate the welding process with this program
i.e. the heat input/distribution. This is done by simply changing the temperature of "the front of the weld spot" to a
specific temperature. The "speed of the spot", "spot dimensions", the "max temperature" and "temperature holding time"
as well as "t8/5", "interpass temperature", "diffusion coefficients" and "convection" are used to calibrate the
simulation.

    speed of the spot: Most likely given by your real experiments weld speed.

    spot dimensions: Most likely also gven by your real experiments parameters.

    max temperature: Could be calculated via energy input during welding but due to thermomechanical uncertainties like
    latent heat (phase change energies), flux/slag etc. thats impractical. Pick for example 1500, or melting temperature
    and adjust "holding time". If for some reason the exact temperature of the weld pool is important, you'll need to
    figure out these uncertainties.

    temperature holding time: This describes how long the simulation forces the "max temperature" on your spot. After this
    time, free diffusion is allowed in your spot. Technically this could also be calculated in concert with "max
    temperature", via your real experiments welding speed but is just as uncertain.

    t8/5: In your real world experiments this results from your process / welding parameters but in the simulation this
    will be used to calibrate

    interpass temperature: In the real welds this is a necessary parameter for the t8/5 / mechanical properties of the
    weld but in the simulation its used to calibrate the heat/temperature field.

    diffusion coefficients: Both for temperature/heat and hydrogen. Obviously you'll need some sort of heat dependent
    hydrogen diffusion parameters if you want to simulate hydrogen diffusion to any degree of accuracy.

    convection: This describes not the "movement" of heat/hydrogen/energy within the weld sample but how they "leave"
    the sample. Think of this as related to boundary conditions. Its important to understand that convection, though
    commonly described as a simple number governing for example the heat exchange of moving air over a metal plate,
    it is exceedingly complicated to accurately assess. In civil engineering for buildings for example, lengthy experiments,
    sometimes over years with dozens of models can be conducted to nail down the convection variable within an order of
    magnitude. This is mainly because the convection variable is basically a geometric variable but is dependant upon things
    like wind speeds and direction, temperature, eddies/vorticies depending on direction etc., surface texture, material
    constants and much much more. For simulations this means that you wont take all of these into consideration and also
    that any number you cleverly come up with holds no meaning once you simplify your simulation for example to 2D or even
    simply change the length of the analyzed/simulated area. This might seem obvious but can also much more insidious
    because, the intuition of even experienced simulators/numerical mathematicions is rooted in reality, where all
    influences happen "automatically" and looking at a larger part of the welding sample does not change phyisics.
    Consider the following:
    A 2D rectangle representing a rod (think of a cigarette made of metal if that helps) is hot on the left
    and room temperature at the right. Now say the heat diffusion is clear within the rod, but we want to include cooling
    at the borders as well. Every step of the simulation some heat is lost, say 1°C per minute or something and you measured
    that in some experiment. If the rod is long enough and you are only interested in the hot area (as you would be, with a
    weld sample, you dont care what happens 100mm away from it) you would limit your simulation to that part, say 50mm away
    from the weld spot. That means you come up with some convection variable, calibrated via temperature of the hot part and
    50mm away which you measured in a real experiment etc. And it works fine, you change the temperature on the left in your
    simulation and in a real experiments and they match. Later you might want to simulate a larger part, maybe 75mm and
    suddenly everything is off. The reason in this particular case is fairly obvious, your convection variable had the
    "invisible part" rolled into it since its a function of the area as well. But there are much more complicated issues
    with this sort of heat/hydrogen exchange. For example the same happens when you change the thickness of your sample
    and use (the most common) temperature diffusion instead of a caloric calculation. This is because of allometric scaling,
    the ratio of volume to surface. The most common application is the Bergmann's rule or polar gigantism. There are many
    more examples that one could come up with: local fluid dynamic differences (hot, turbulent etc.), phase changes,
    extreme heat - radiation etc.
    The important thing to remember is that what we (mostly for convenience) call "convection" here is simply the additional
    loss of some hydrogen/heat at the bounadry or over the surface due to some real world phenomenae which we wont be
    simulating. This necessitates an adjustment or at the very least control of these numbers when ever we change the
    simulation, since it is used to calibrate the temperature at some known/measured points.

If everything is done properly, you now have the heat calibrated via the interpass temperature at some spot you
measured, the t8/5 at the weld bead, the boundary conditions and the convection. If thee heat/temperature fits at these
spots the simulation will then "interpolate" the heat field everywhere else.

Now you have the simulation basically done. Next ask yourself what you actually want to find out. For this example,
we wanted to know how different DHT temperatures/times effect the distribution and we also wanted to look at the
effect of plate thickness.

... Continue later.. you're still sick...

"""