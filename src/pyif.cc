// for now, only pyif.cc #include's Python.h and the numpy.
// this may certainly be too limiting in the long run and isn't neccessary by any means. 
// for python, we would move the python include from here to boda_tu_base.H if we want to use the Python/C-API globally.
// for numpy, if we wish to include numpy in other TUs (.cc files / objects), we need to do some magic: all other files must #define the same PY_ARRAY_UNIQUE_SYMBOL, and all but this one (that calls _import_array(), must #define NO_IMPORT_ARRAY
#include<Python.h>
#include"boda_tu_base.H"
#define PY_ARRAY_UNIQUE_SYMBOL boda_numpy_unique_symbol_yo
#include<numpy/arrayobject.h>
#include"pyif.H"

namespace boda 
{
  using namespace std;

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

  void py_init( char const * const prog_name )
  {
    Py_SetProgramName((char *)prog_name);
    Py_Initialize();
    ppyo main_mod( PyImport_AddModule( "__main__" ), 1 );
    ppyo main_dict( PyModule_GetDict( main_mod.get() ), 1 );
    ppyo ret( PyRun_String("import sys,os.path;"
			   " sys.path.append(os.path.join(os.path.split("
			   "os.readlink('/proc/self/exe'))[0],'..','pysrc'))",
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
  

  void prc_plot( std::string const & class_name, uint32_t const tot_num_class, vect_prc_elem_t const & prc_elems )
  {
    ppyo module = import_module( "bplot" );
    ppyo func( get_callable( module, "plot_stuff" ) );
    npy_intp dims[2] = {3,prc_elems.size()};
    ppyo args( PyTuple_New(2) );
    ppyo npa( PyArray_SimpleNew( 2, dims, NPY_DOUBLE) );
    // fill in npa
    uint32_t ix = 0;
    for ( vect_prc_elem_t::const_iterator i = prc_elems.begin(); i != prc_elems.end(); ++i)
    {
      *((double *)PyArray_GETPTR2( npa.get(), 0, ix )) = i->get_recall( tot_num_class );
      *((double *)PyArray_GETPTR2( npa.get(), 1, ix )) = i->get_precision();
      *((double *)PyArray_GETPTR2( npa.get(), 2, ix )) = i->score;
      ++ix;
    }
    tuple_set( args, 0, ppyo(PyString_FromString(class_name.c_str())) );
    tuple_set( args, 1, npa );
    ppyo ret = call_obj( func, args );
    printf("Result of call: %ld\n", PyInt_AsLong(ret.get()));
  }

}
