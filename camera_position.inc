// POV-Ray 3.6 / 3.7 Scene File "Ribbon_Cable_1.pov"
// author: Friedrich A. Lohmueller, Sept-2009/Jan-2011
// email: Friedrich.Lohmueller_at_t-online.de
// homepage: http://www.f-lohmueller.de
//--------------------------------------------------------------------------
#version 3.6; // 3.7;
global_settings{ assumed_gamma 1.0 }
#default{ finish{ ambient 0.1 diffuse 0.9 }}
//--------------------------------------------------------------------------
#include "colors.inc"
#include "textures.inc"
#include "glass.inc"
#include "metals.inc"
#include "golds.inc"
#include "stones.inc"
#include "woods.inc"
#include "shapes.inc"
#include "shapes2.inc"
#include "functions.inc"
#include "math.inc"
#include "transforms.inc"
//--------------------------------------------------------------------------------------------------------<<<<
//------------------------------------------------------------- Camera_Position, Camera_look_at, Camera_Angle
	#declare Camera_Number = 21;
//----------------------------------------89----------------------------------------------------------------<<<<
#switch ( Camera_Number )
#case (15) // 3D reaching arm
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <50.00, 10.00, -60.00> ;  // Topside view
  #declare Camera_Look_At  = < -3.50, 5.700,  10.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#break
#case (16) // obstacle nest
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <50.00, 50.00, -70.00> ;  // Topside view
  #declare Camera_Look_At  = < -13.50, -15.700,  50.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (17) // obstacle maze
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <80.00, 20.00, -20.00> ;  // Topside view
  #declare Camera_Look_At  = < -13.50, 5.700,  20.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (18) // arm reaching with orientation #1
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <55.00, 20.00, -50.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.00, 4.20, 5.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#case (19) // arm reaching with orientation #2
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <50.00, 17.00, -60.00> ;  // Topside view
  #declare Camera_Look_At  = < -3.50, 5.700, 10.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#break
#case (20) // arm reaching with orientation #3 -- zoomed in
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <30.00, 15.00, -40.00> ;  // Topside view
  #declare Camera_Look_At  = < -3.50, 5.700, 10.00> ;
  #declare Camera_Angle    =  30 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#break
#case (21) // PICK-AND-PLACE (Case_2 posrelease): sees arm + object + drop-off
  // Cases 15-20 were all framed for the original REACHING task: the camera sits at
  // +X/-Z looking toward -X/+Z, which is within 6-24 deg of the arm-base -> drop-off
  // line, so the phase-2 drop-off marker renders directly BEHIND the rod and the
  // place/release action is invisible. This case views that line side-on instead:
  // the view axis is 90 deg to base->drop-off, so base, carried object and drop-off
  // spread across the frame.
  //
  // Derived, not eyeballed. In POVray coords (sim -> POV is X=2-12x, Y=12y, Z=4-12z
  // via generate_POVray_files.py scale=3 + "scale<4,4,4> rotate<0,90,90> translate<2,0,4>"):
  //   arm base <2, 0.3, 4>   drop-off <-2.8, 0.6, 8.8>   pick targets Y up to ~10.8
  // Position is perpendicular to base->drop-off and elevated; Look_At/Angle were fitted
  // so every rod node, the sphere AND the director arrow tips stay inside the frame
  // across all episode frames (POVray "angle" is HORIZONTAL, so at 16:9 the vertical
  // half-FOV is only ~0.56x of it -- that is what clipped the tip triad at angle 30).
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <42.00, 20.00, 48.00> ;
  #declare Camera_Look_At  = <-1.42, 6.17, 5.65> ;
  #declare Camera_Angle    =  31.4 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#else  // default
  #declare Ultra_Wide_Angle_On = 0;// don't use fish eye - for proper text
  #declare Camera_Position = <23.00, 6.00,-18.00> ;  // Topside view
  #declare Camera_Look_At  = < 0.50, 1.700,  7.00> ;
  #declare Camera_Angle    =  28 ;
  #declare Camera_Rotate = <0,0,0>; // tilling of the camera!!
#break
#end // of "#switch ( Camera_Number )" -----------------------------
//---------------------------------------------------------------------

camera{ location Camera_Position
        right    x*image_width/image_height
        angle    Camera_Angle
        look_at  Camera_Look_At
      }
//light_source{< 0,300,500> color White}
light_source{< 1500,2500,-1000>color White}           // sun light
light_source{ Camera_Position  color rgb<0.9,0.9,1>*0.1}  // flash light


/*
// fog ---------------------------------------------------------------
fog{fog_type   2   distance 65  color rgb<1,1,1>
fog_offset 0.1 fog_alt  2.0 turbulence 0.2}
//--------------------------------------------------------------------
*/
// <1,0,0>, -3
// <0,1,0>, 0
//<0,0,1>,-200
background { color White }

plane { <0,1.5,0>,-0.1  // plane with layered textures
texture { pigment{color White*1.1}
             }
}

//finish {ambient 0.45 diffuse 0.85}
/*
//------------------------------ the Axes --------------------------------
//------------------------------------------------------------------------
#macro Axis_( AxisLen, Dark_Texture,Light_Texture)
 union{
    cylinder { <0,-AxisLen,0>,<0,AxisLen,0>,0.05
               texture{checker texture{Dark_Texture }
                               texture{Light_Texture}
               scale 0.5
               translate<0.1,0,0.1>}
             }
    cone{<0,AxisLen,0>,0.2,<0,AxisLen+0.7,0>,0
          texture{Dark_Texture}
         }
     } // end of union
#end // of macro "Axis()"
//------------------------------------------------------------------------
#macro AxisXYZ( AxisLenX, AxisLenY, AxisLenZ, Tex_Dark, Tex_Light)
//--------------------- drawing of 3 Axes --------------------------------
union{
#if (AxisLenX != 0)
 object { Axis_(AxisLenX, Tex_Dark, Tex_Light)   rotate< 0,0,-90>}// x-Axis
#end // of #if
#if (AxisLenY != 0)
 object { Axis_(AxisLenY, Tex_Dark, Tex_Light)   rotate< 0,0,  0>}// y-Axis
#end // of #if
#if (AxisLenZ != 0)
 object { Axis_(AxisLenZ, Tex_Dark, Tex_Light)   rotate<90,0,  0>}// z-Axis
#end // of #if
} // end of union
#end// of macro "AxisXYZ( ... )"
//------------------------------------------------------------------------

#declare Texture_A_Dark  = texture {
                               pigment{ color rgb<1,0.4,0>}
                               finish { phong 1}
                             }
#declare Texture_A_Light = texture {
                               pigment{ color rgb<1,1,1>}
                               finish { phong 1}
                             }

object{ AxisXYZ( 2.70, 2.70, 2.70, Texture_A_Dark, Texture_A_Light) scale 0.75 }
//-------------------------------------------------- end of coordinate axes


// ground -----------------------------------------------------------------
//---------------------------------<<< settings of squared plane dimensions
#declare RasterScale = 2.0;
#declare RasterHalfLine  = 0.03;
#declare RasterHalfLineZ = 0.03;
//-------------------------------------------------------------------------
#macro Raster(RScale, HLine)
       pigment{ gradient x scale RScale
                color_map{[0.000   color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,0>*0.6]
                          [0+HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,1>]
                          [1-HLine color rgbt<1,1,1,0>*0.6]
                          [1.000   color rgbt<1,1,1,0>*0.6]} }
 #end// of Raster(RScale, HLine)-macro
//-------------------------------------------------------------------------


plane { <0,1,0>, 0    // plane with layered textures
        texture { pigment{color White*1.1}
                  finish {ambient 0.45 diffuse 0.85}}
        texture { Raster(RasterScale,RasterHalfLine ) rotate<0,0,0> }
        texture { Raster(RasterScale,RasterHalfLineZ) rotate<0,90,0>}
        rotate<0,0,0>
      }
      */
//------------------------------------------------ end of squared plane XZ
//------------------------------------------------------------------------------------------------------- 
//------------------------------------------------------------------------------------------------------- 
//--------------------------------------------------------------------- macro "Vector(Start,End,Radius)"!
#macro Vector(P_start,P_end, R_Vector)  
union{

cylinder{ P_start, P_end - ( vnormalize(P_end - P_start)*9.5*R_Vector), R_Vector  }
cone    { P_end - ( vnormalize(P_end - P_start)*10*R_Vector), 3*R_Vector, P_end, 0 }
}// end of union
#end //------------------------------------------------------------------------------------- end of macro
// alternatively:   #include "analytical_g.inc"
//-------------------------------------------------------------------------------------------------------- 
//--------------------------------------------------------------------------
//---------------------------- objects in scene ----------------------------
//--------------------------------------------------------------------------
