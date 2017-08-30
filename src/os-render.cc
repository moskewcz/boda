// Copyright (c) 2015, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
#include"boda_tu_base.H"
#include"os-render.H"
#include"img_io.H"
#include"has_main.H"
#include"str_util.H"

#include"GL/osmesa.h"
#include"GL/glu.h"


namespace boda 
{

  void Sphere(float radius, int slices, int stacks)
  {
    GLUquadric *q = gluNewQuadric();
    gluQuadricNormals(q, GLU_SMOOTH);
    gluSphere(q, radius, slices, stacks);
    gluDeleteQuadric(q);
  }

  
  struct data_to_img_pts_t : virtual public nesi, public data_stream_t // NESI(help="annotate data blocks (containing point cloud data) with image representations (in as_img field of data block). returns annotated data block.",
                           // bases=["data_stream_t"], type_id="add-img-pts")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    u32_pt_t disp_sz; //NESI(default="300:300",help="X/Y per-stream-image size")

    p_img_t frame_buf;

    OSMesaContext ctx;

    virtual void data_stream_init( nesi_init_arg_t * const nia ) {
      frame_buf = make_shared< img_t >();
      frame_buf->set_sz_and_alloc_pels( disp_sz );
      ctx = OSMesaCreateContextExt( OSMESA_RGBA, 16, 0, 0, NULL );
      if (!ctx) { rt_err("OSMesaCreateContext failed!"); }
      
      if (!OSMesaMakeCurrent( ctx, frame_buf->get_row_addr(0), GL_UNSIGNED_BYTE, frame_buf->sz.d[0], frame_buf->sz.d[1] )) {
        rt_err("OSMesaMakeCurrent failed.\n");
      }
      {
        int z, s, a;
        glGetIntegerv(GL_DEPTH_BITS, &z);
        glGetIntegerv(GL_STENCIL_BITS, &s);
        glGetIntegerv(GL_ACCUM_RED_BITS, &a);
        printf("Depth=%d Stencil=%d Accum=%d\n", z, s, a);
      }
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

      GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
      GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
      GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
      GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
      GLfloat red_mat[]   = { 1.0, 0.2, 0.2, 1.0 };
      GLfloat green_mat[] = { 0.2, 1.0, 0.2, 1.0 };
      GLfloat blue_mat[]  = { 0.2, 0.2, 1.0, 1.0 };


      glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
      glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
      glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
      glLightfv(GL_LIGHT0, GL_POSITION, light_position);

      glEnable(GL_LIGHTING);
      glEnable(GL_LIGHT0);
      glEnable(GL_DEPTH_TEST);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(-2.5, 2.5, -2.5, 2.5, -10.0, 10.0);
      glMatrixMode(GL_MODELVIEW);

      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

      glPushMatrix();
      glRotatef(20.0, 1.0, 0.0, 0.0);
#if 0
      glPushMatrix();
      glTranslatef(-0.75, 0.5, 0.0);
      glRotatef(90.0, 1.0, 0.0, 0.0);
      glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, red_mat );
      Torus(0.275, 0.85, 20, 20);
      glPopMatrix();

      glPushMatrix();
      glTranslatef(-0.75, -0.5, 0.0);
      glRotatef(270.0, 1.0, 0.0, 0.0);
      glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, green_mat );
      Cone(1.0, 2.0, 16, 1);
      glPopMatrix();
#endif
      glPushMatrix();
      glTranslatef(0.75, 0.0, -1.0);
      glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue_mat );
      Sphere(1.0, 20, 20);
      glPopMatrix();

      glPopMatrix();

      /* This is very important!!!
       * Make sure buffered commands are finished!!!
       */
      glFinish();

      
    }


    virtual string get_pos_info_str( void ) {      
      return strprintf( "data-to-img: disp_sz=%s", str(disp_sz).c_str() );
    }

    
  };

#include"gen/os-render.cc.nesi_gen.cc"

}
