SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}  -ltcmalloc ")
#SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}  -ltcmalloc_minimal ")

# uncomment one to configure libhand depend.
SET(DD_ENABLE_HAND_SYNTH 1)
#unset(DD_ENABLE_HAND_SYNTH)

# uncomment one to configure OpenNI depend.
SET(DD_ENABLE_OPENNI 1)
#unset(DD_ENABLE_OPENNI)


