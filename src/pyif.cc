// Copyright (c) 2013-2014, Matthew W. Moskewicz <moskewcz@alumni.princeton.edu>; part of Boda framework; see LICENSE
// for now, only pyif.cc #include's Python.h and the numpy.
// this may certainly be too limiting in the long run and isn't neccessary by any means. 
// for python, we would move the python include from here to boda_tu_base.H if we want to use the Python/C-API globally.
// for numpy, if we wish to include numpy in other TUs (.cc files / objects), we need to do some magic: all other files must #define the same PY_ARRAY_UNIQUE_SYMBOL, and all but this one (that calls _import_array(), must #define NO_IMPORT_ARRAY
#include<Python.h>
#include"boda_tu_base.H"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL boda_numpy_unique_symbol_yo
#include<numpy/arrayobject.h>
#include"pyif.H"
#include"img_io.H"
#include"results_io.H"
#include<boost/filesystem.hpp>

namespace boda 
{
  using boost::filesystem::path;
  using boost::filesystem::canonical;

  void rt_py_err( std::string const & err_msg ) {
    PyErr_Print(); 
    // we don't just call rt_err() here so we can keep the stack skip
    // depths of rt_py_err() and rt_err() the same
    throw rt_exception( "error: " + err_msg, get_backtrace() ); 
  }

  // simple smart pointer for PyObject *
  struct ppyo
  {
    PyObject * p;
    ppyo( void ) : p(0) { }
    ~ppyo( void ) { Py_XDECREF(p); }
    ppyo & operator = ( ppyo const & r )
    {
      Py_XDECREF( p );
      Py_XINCREF( r.p );
      p = r.p;
      return *this;
    }
    ppyo( ppyo const & r )
    {
      p = r.p;
      Py_XINCREF( r.p );
    }
    // assume null is an error, and that a generic error message is diagnostic enough (with the stack traces)
    ppyo( PyObject * const p_, bool const is_borrowed=0 ) : p(p_) 
    {
      if( !p ) { rt_py_err( "null PyObject" ); }
      if( is_borrowed ) { Py_XINCREF( p ); }
    }
    PyObject * get( void ) const { return p; }
  };
  typedef vector< ppyo > vect_ppyo;

  void py_init( char const * const prog_name )
  {
    string const boda_pysrc_dir = canonical(path(py_boda_dir())/"pysrc").string();
    Py_SetProgramName((char *)prog_name);
    Py_Initialize();
    ppyo main_mod( PyImport_AddModule( "__main__" ), 1 );
    ppyo main_dict( PyModule_GetDict( main_mod.get() ), 1 );
    ppyo ret( PyRun_String( ("import sys,os.path,signal;"
			     "boda_dir = '"+py_boda_dir()+"';"
			     "boda_test_dir = '"+py_boda_test_dir()+"';"
			     " sys.path.append('"+boda_pysrc_dir+"'); signal.signal(signal.SIGINT, signal.SIG_DFL);").c_str(),
			   Py_file_input, main_dict.get(), main_dict.get() ) );
    if( _import_array() < 0 ) { rt_err( "failed to import numpy" ); }

  }


  void py_finalize( void )
  {
    Py_Finalize();
  }

  ppyo import_module( string const & module_name )
  {
    ppyo pName( PyString_FromString(module_name.c_str()) );
    ppyo pModule( PyImport_Import(pName.get()) );
    return pModule;
  }

  ppyo get_callable( ppyo obj, string const & attr )
  {
    ppyo pFunc( PyObject_GetAttrString(obj.get(), attr.c_str()) );
    if( !PyCallable_Check(pFunc.get()) ) { rt_err( "attr '"+attr+"' not callable" ); }
    return pFunc;
  }

  ppyo call_obj( ppyo obj, ppyo args )
  {
    ppyo ret( PyObject_CallObject(obj.get(), args.get()) );
    return ret;
  }

  void tuple_set( ppyo const & t, uint32_t const ix, ppyo const & v )
  {
    Py_INCREF( v.get() );
    int const ret = PyTuple_SetItem( t.get(), ix, v.get() );
    if( ret ) { rt_err( "tuple setitem failed" ); }
  }
  
  ppyo bplot_call( char const * const fn, vect_ppyo const & args )
  {
    ppyo module = import_module( "bplot" );
    ppyo func( get_callable( module, fn ) );
    ppyo py_args( PyTuple_New(args.size()) );
    for( uint32_t arg_ix = 0; arg_ix < args.size(); ++arg_ix ) {
      tuple_set( py_args, arg_ix, args[arg_ix] );
    }
    return call_obj( func, py_args );
  }
  ppyo bplot_call( char const * const fn, ppyo arg ) { 
    vect_ppyo args; args.push_back( arg ); return bplot_call( fn, args ); }
  ppyo bplot_call( char const * const fn, ppyo arg1, ppyo arg2 ) {
    vect_ppyo args; args.push_back( arg1 ); args.push_back( arg2 ); return bplot_call( fn, args ); }
  ppyo bplot_call( char const * const fn, ppyo arg1, ppyo arg2, ppyo arg3 ) {
    vect_ppyo args; args.push_back( arg1 ); args.push_back( arg2 ); args.push_back( arg3); 
    return bplot_call( fn, args ); }

  void prc_plot( std::string const & plt_fn, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems,
		 std::string const & plt_title )
  {
    npy_intp dims[2] = {3,(npy_intp)prc_elems.size()};
    PyArrayObject * raw_npa = (PyArrayObject*)PyArray_SimpleNew( 2, dims, NPY_DOUBLE);
    ppyo npa( (PyObject*)raw_npa );
    
    uint32_t ix = 0;
    for ( vect_prc_elem_t::const_iterator i = prc_elems.begin(); i != prc_elems.end(); ++i)
    {
      *((double *)PyArray_GETPTR2( raw_npa, 0, ix )) = i->get_recall( tot_num_class );
      *((double *)PyArray_GETPTR2( raw_npa, 1, ix )) = i->get_precision();
      *((double *)PyArray_GETPTR2( raw_npa, 2, ix )) = i->score;
      ++ix;
    }
    ppyo ret = bplot_call( "plot_stuff", ppyo(PyString_FromString(plt_fn.c_str())), npa,
			   ppyo(PyString_FromString(plt_title.c_str())) );
    //printf("Result of call: %ld\n", PyInt_AsLong(ret.get()));
  }

  ppyo img_to_py( p_img_t img )
  {
    npy_intp dims[3] = {img->sz.d[1],img->sz.d[0],img->depth};
#if 0 // for reference, if we wanted to copy into a newly allocated numpy array
    ppyo npa( PyArray_SimpleNew( 3, dims, NPY_UINT8) );
    for( uint32_t y = 0; y < img->h; ++y ) {
      for( uint32_t x = 0; x < img->w; ++x ) {
	for( uint32_t c = 0; c < img->depth; ++c ) {
	  *((uint8_t *)PyArray_GETPTR3( npa.get(), y, x, c )) = img->pels.get()[y*img->row_pitch + x*img->depth + c];
	}
      }
    }
#else
    npy_intp strides[3] = {img->row_pitch,img->depth,1}; 
    // FIXME: untracked reference to img->pels() taken here, so img must outlive npa. fixable? hold npa in img?
    return ppyo( PyArray_New( &PyArray_Type, 3, dims, NPY_UINT8, strides, img->pels.get(), 0, 0, 0 ) );
#endif
  }

  void py_img_show( p_img_t img, string const & save_as_filename )
  {
    bplot_call( "img_show", img_to_py( img ), ppyo(PyString_FromString(save_as_filename.c_str())) );
  }


  void show_dets( p_img_t img, vect_base_scored_det_t const & scored_dets ) {
    npy_intp dims[2] = {(npy_intp)scored_dets.size(),4};
    ppyo npa( PyArray_SimpleNew( 2, dims, NPY_UINT32) );
    for( vect_base_scored_det_t::const_iterator i = scored_dets.begin(); i != scored_dets.end(); ++i ) {
      for( uint32_t d = 0; d < 4; ++d ) {
	*((uint32_t *)PyArray_GETPTR2( (PyArrayObject*)npa.get(), i-scored_dets.begin(), d )) = i->p[d>>1].d[d&1];
      }
    }
    bplot_call( "show_dets", img_to_py(img), npa );
  }

}
