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

  static void
  Cone(float base, float height, int slices, int stacks)
  {
    GLUquadric *q = gluNewQuadric();
    gluQuadricDrawStyle(q, GLU_FILL);
    gluQuadricNormals(q, GLU_SMOOTH);
    gluCylinder(q, base, 0.0, height, slices, stacks);
    gluDeleteQuadric(q);
  }


  static void
  Torus(float innerRadius, float outerRadius, int sides, int rings)
  {
    /* from GLUT... */
    int i, j;
    GLfloat theta, phi, theta1;
    GLfloat cosTheta, sinTheta;
    GLfloat cosTheta1, sinTheta1;
    const GLfloat ringDelta = 2.0 * M_PI / rings;
    const GLfloat sideDelta = 2.0 * M_PI / sides;

    theta = 0.0;
    cosTheta = 1.0;
    sinTheta = 0.0;
    for (i = rings - 1; i >= 0; i--) {
      theta1 = theta + ringDelta;
      cosTheta1 = cos(theta1);
      sinTheta1 = sin(theta1);
      glBegin(GL_QUAD_STRIP);
      phi = 0.0;
      for (j = sides; j >= 0; j--) {
        GLfloat cosPhi, sinPhi, dist;

        phi += sideDelta;
        cosPhi = cos(phi);
        sinPhi = sin(phi);
        dist = outerRadius + innerRadius * cosPhi;

        glNormal3f(cosTheta1 * cosPhi, -sinTheta1 * cosPhi, sinPhi);
        glVertex3f(cosTheta1 * dist, -sinTheta1 * dist, innerRadius * sinPhi);
        glNormal3f(cosTheta * cosPhi, -sinTheta * cosPhi, sinPhi);
        glVertex3f(cosTheta * dist, -sinTheta * dist,  innerRadius * sinPhi);
      }
      glEnd();
      theta = theta1;
      cosTheta = cosTheta1;
      sinTheta = sinTheta1;
    }
  }

  
  struct data_to_img_pts_t : virtual public nesi, public data_stream_t // NESI(help="annotate data blocks (containing point cloud data) with image representations (in as_img field of data block). returns annotated data block.",
                           // bases=["data_stream_t"], type_id="add-img-pts")
  {
    virtual cinfo_t const * get_cinfo( void ) const; // required declaration for NESI support
    uint32_t verbose; //NESI(default="0",help="verbosity level (max 99)")
    u32_pt_t disp_sz; //NESI(default="300:300",help="X/Y per-stream-image size")

    p_img_t frame_buf;

    OSMesaContext ctx;

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
      //glOrtho(-2.5, 2.5, -2.5, 2.5, -10.0, 10.0);
      gluPerspective( 90, double( frame_buf->sz.d[0] ) / double( frame_buf->sz.d[1] ), .1, 1000 );
      glMatrixMode(GL_MODELVIEW);

      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

      glLoadIdentity();
//      glRotatef(20.0, 1.0, 0.0, 0.0);
      glRotatef( -cam_rot[2], 0.0f, 0.0f, 1.0f);
      glRotatef( -cam_rot[1], 0.0f, 1.0f, 0.0f);
      glRotatef( -cam_rot[0], 1.0f, 0.0f, 0.0f);
      glTranslatef( -cam_pos[0], -cam_pos[1], -cam_pos[2] );

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

      glPushMatrix();
      glTranslatef(0.75, 0.0, -1.0);
      glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, blue_mat );
      Sphere(1.0, 20, 20);
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
