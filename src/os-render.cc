// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"os-render.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"

#include"GL/glew.h"
#define GLAPI extern 
#include"GL/osmesa.h"

#include <glm/glm.hpp>
using namespace glm;

#include"ext/shader.hpp"


namespace boda 
{

  struct data_to_img_pts_t : virtual public nesi, public data_stream_t // NESI(help="annotate data blocks (containing point cloud data) with image representations (in as_img field of data block). returns annotated data block.",
                           // bases=["data_stream_t"], type_id="add-img-pts")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    u32_pt_t disp_sz; //NESI(default="300:300",help="X/Y per-stream-image size")

    p_img_t frame_buf;

    OSMesaContext ctx;
    GLuint programID;
    GLuint vertexbuffer;
    
    float cam_pos[3];
    float cam_rot[3];
    
    virtual void set_opt( data_stream_opt_t const & opt ) {
      if( opt.name == "camera-pos-rot" ) {
        if( (!opt.val.get()) || (opt.val->dims.tn != "float") || (opt.val->elems_sz() != 6) ) {
          rt_err( "add-img-pts: couldn't parse camera-pos opt" );
        }
        nda_T<float> cam_pos_rot( opt.val );
        for( uint32_t i = 0; i != 3; ++i ) {
          cam_pos[i] = cam_pos_rot.at2(0,i);
          cam_rot[i] = cam_pos_rot.at2(1,i);
        }
        printf( "cam_pos[0]=%s cam_pos[1]=%s cam_pos[2]=%s\n", str(cam_pos[0]).c_str(), str(cam_pos[1]).c_str(), str(cam_pos[2]).c_str() );
      }
      
    }

    
    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      for( uint32_t i = 0; i != 3; ++i ) { cam_pos[i] = 0.0f; cam_rot[i] = 0.0f; }
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( disp_sz );
      int const osmesa_attrs[] = {
        OSMESA_FORMAT, OSMESA_RGBA,
        OSMESA_DEPTH_BITS, 16,
        OSMESA_STENCIL_BITS, 0,
        OSMESA_ACCUM_BITS, 0,
        OSMESA_PROFILE, OSMESA_CORE_PROFILE,
        OSMESA_CONTEXT_MAJOR_VERSION, 3,
        OSMESA_CONTEXT_MINOR_VERSION, 3,
        0 };
    
//      ctx = OSMesaCreateContextExt( OSMESA_RGBA, 16, 0, 0, NULL );
       ctx = OSMesaCreateContextAttribs( osmesa_attrs, NULL );
      if (!ctx) { rt_err("OSMesaCreateContext failed!"); }
      
      if (!OSMesaMakeCurrent( ctx, frame_buf->get_row_addr(0), GL_UNSIGNED_BYTE, frame_buf->sz.d[0], frame_buf->sz.d[1] )) {
        rt_err("OSMesaMakeCurrent failed.\n");
      }

      // Initialize GLEW
      glewExperimental = true; // Needed for core profile
      if (glewInit() != GLEW_OK) { rt_err( "Failed to initialize GLEW" ); }

      
      {
        int z, s, a;
        glGetIntegerv(GL_DEPTH_BITS, &z);
        glGetIntegerv(GL_STENCIL_BITS, &s);
        glGetIntegerv(GL_ACCUM_RED_BITS, &a);
        printf("Depth=%d Stencil=%d Accum=%d\n", z, s, a);
        printf( "GL_VERSION: %s\nGL_VENDOR: %s\nGL_RENDERER: %s\n", glGetString(GL_VERSION), glGetString(GL_VENDOR), glGetString(GL_RENDERER) );
        
      }

      // Dark blue background
      glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

      GLuint VertexArrayID;
      printf( "PREVertexArrayID=%s\n", str(VertexArrayID).c_str() );
      glGenVertexArrays(1, &VertexArrayID);
      printf( "PRE2VertexArrayID=%s\n", str(VertexArrayID).c_str() );
      glBindVertexArray(VertexArrayID);
      printf( "VertexArrayID=%s\n", str(VertexArrayID).c_str() );
      
      // Create and compile our GLSL program from the shaders
      programID = LoadShaders( "SimpleVertexShader.vertexshader", "SimpleFragmentShader.fragmentshader" );
      if( programID == 0 ) { rt_err("loading shaders failed, files not found."); }
      printf( "programID=%s\n", str(programID).c_str() );

      static const GLfloat g_vertex_buffer_data[] = { 
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
      };

      glGenBuffers(1, &vertexbuffer);
      glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
      glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

      
    }

    virtual data_block_t proc_block( data_block_t const & db ) {
      if( !db.nda.get() ) { rt_err( "add-img-pts: expected nda data in block, but found none." ); }
      render_pts_into_frame_buf( db );
      data_block_t ret = db;
      ret.as_img = frame_buf;
      return ret;
    }

    virtual void render_pts_into_frame_buf( data_block_t const & db ) {
      p_nda_t const & nda = db.nda;


      // Clear the screen
      glClear( GL_COLOR_BUFFER_BIT );

      // Use our shader
      glUseProgram(programID);

      // 1rst attribute buffer : vertices
      glEnableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
      glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
                            );

      // Draw the triangle !
      glDrawArrays(GL_TRIANGLES, 0, 3); // 3 indices starting at 0 -> 1 triangle

      glDisableVertexAttribArray(0);
      glFinish();

#if 0                
	// Cleanup VBO
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteVertexArrays(1, &VertexArrayID);
	glDeleteProgram(programID);

      
#endif

      
    }


    virtual string get_pos_info_str( void ) {      
      return strprintf( "data-to-img: disp_sz=%s", str(disp_sz).c_str() );
    }

    
  };

#include"gen/os-render.cc.nesi_gen.cc"

}
